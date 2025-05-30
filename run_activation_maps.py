import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import argparse
import os
import torch.nn as nn
import cv2
import numpy as np
from torch.autograd import Variable
from helpers.lr_schedule import exp_warmup_linear_down
from helpers.init import worker_init_fn
from models.passt import get_model
from models.mel import AugmentMelSTFT
from helpers import nessi
from datasets.dcase23_dev import get_training_set, get_test_set, get_eval_set
from helpers.utils import mixstyle, mixup_data
import json
from pytorch_grad_cam import GradCAM, HiResCAM, GradCAMPlusPlus, DeepFeatureFactorization, GradCAMTrans
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


torch.set_float32_matmul_precision("highest")
# def load_and_modify_checkpoint(pl_module,num_classes=10):
#         # Modify the final layer
#         pl_module.model.head = nn.Sequential(
#             nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True),
#             nn.Linear(768, num_classes)
#         )
#         return pl_module
class PLModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        # model to preprocess waveforms into log mel spectrograms
        self.mel = AugmentMelSTFT(n_mels=config.n_mels,
                                  sr=config.resample_rate,
                                  win_length=config.window_size,
                                  hopsize=config.hop_size,
                                  n_fft=config.n_fft,
                                  freqm=config.freqm,
                                  timem=config.timem,
                                  fmin=config.fmin,
                                  fmax=config.fmax,
                                  fmin_aug_range=config.fmin_aug_range,
                                  fmax_aug_range=config.fmax_aug_range
                                  )

        self.model = get_model(arch="passt_s_swa_p16_128_ap476",
                             n_classes=config.n_classes,
                             input_fdim=config.input_fdim,
                             s_patchout_t=config.s_patchout_t,
                             s_patchout_f=config.s_patchout_f)

        self.device_ids = ['a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']
        self.label_ids = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square', 'shopping_mall',
                          'street_pedestrian', 'street_traffic', 'tram']
        self.device_groups = {'a': "real", 'b': "real", 'c': "real",
                              's1': "seen", 's2': "seen", 's3': "seen",
                              's4': "unseen", 's5': "unseen", 's6': "unseen"}

        self.calc_device_info = True
        self.epoch = 0
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.aug_smooth=config.aug_smooth
        self.epoch_activation_maps = []
    def reshape_transform(self,tensor, height=16, width=16):
        # print(f"reshapping a tensor of size: {tensor.shape}")
        # Remove both class token and distillation token
        tensor = tensor[:, 2:, :]  # Exclude the first two tokens (cls and dist)
        
        # Define the reshaping dimensions based on 108 remaining tokens
        height, width = 12, 9
        result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
        
        # Transpose to make it [Batch, Channels, Height, Width]
        result = result.transpose(2, 3).transpose(1, 2)
        return result
    
    def mel_forward(self, x):
        old_shape = x.size()
        x = x.reshape(-1, old_shape[2])
        x = self.mel(x)
        x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
        return x

    def forward(self, x):
        return self.model(x)
    def forward_cam(self,x):
        x = self.mel_forward(x)
        x = self.model(x)
        return x
    
    def predict_step(self, eval_batch, batch_idx, dataloader_idx=0):
        x, files = eval_batch
        x = self.mel_forward(x)
        # x = x.half()
        y_hat, embed = self.forward(x)

        return files, y_hat
    # def mixup_criterion(self,criterion, pred, y_a, y_b, lam):
    #         return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
        


    def validation_step(self, batch, batch_idx):
        x, files, labels, devices, cities = batch

        if self.mel:
            x = self.mel_forward(x)

        y_hat, embed = self.forward(x)
        labels = labels.long()
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")
        loss = samples_loss.mean()

        self.log("validation.loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        _, preds = torch.max(y_hat, dim=1)
        n_correct_pred_per_sample = (preds == labels)
        n_correct_pred = n_correct_pred_per_sample.sum()
        results = {"val_loss": loss, "n_correct_pred": n_correct_pred, "n_pred": len(labels)}

        '''if self.calc_device_info:
            devices = [d.rsplit("-", 1)[1][:-4] for d in files]
            for d in self.device_ids:
                results["devloss." + d] = torch.as_tensor(0., device=self.device)
                results["devcnt." + d] = torch.as_tensor(0., device=self.device)
                results["devn_correct." + d] = torch.as_tensor(0., device=self.device)

            for i, d in enumerate(devices):
                results["devloss." + d] = results["devloss." + d] + samples_loss[i]
                results["devn_correct." + d] = results["devn_correct." + d] + n_correct_pred_per_sample[i]
                results["devcnt." + d] = results["devcnt." + d] + 1'''
        return results

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        val_acc = sum([x['n_correct_pred'] for x in outputs]) * 1.0 / sum(x['n_pred'] for x in outputs)
        logs = {'val.loss': avg_loss, 'val_acc': val_acc}

        '''if self.calc_device_info:
            for d in self.device_ids:
                dev_loss = torch.stack([x["devloss." + d] for x in outputs]).sum()
                dev_cnt = torch.stack([x["devcnt." + d] for x in outputs]).sum()
                dev_corrct = torch.stack([x["devn_correct." + d] for x in outputs]).sum()
                logs["vloss." + d] = dev_loss / dev_cnt
                logs["vacc." + d] = dev_corrct / dev_cnt
                logs["vcnt." + d] = dev_cnt
                # device groups
                logs["acc." + self.device_groups[d]] = logs.get("acc." + self.device_groups[d], 0.) + dev_corrct
                logs["count." + self.device_groups[d]] = logs.get("count." + self.device_groups[d], 0.) + dev_cnt
                logs["lloss." + self.device_groups[d]] = logs.get("lloss." + self.device_groups[d], 0.) + dev_loss

            for d in set(self.device_groups.values()):
                logs["acc." + d] = logs["acc." + d] / logs["count." + d]
                logs["lloss.False" + d] = logs["lloss." + d] / logs["count." + d]'''

        self.log_dict(logs)

        if self.epoch > 0:
            print()
            print(f"Validation Loss: {avg_loss}")
            print(f"Validation Accuracy: {val_acc}")

        self.epoch += 1
    def test_step(self, test_batch, batch_idx):
        x, files, labels, devices, cities = test_batch
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device=x.device)
        # maximum memory allowance for parameters: 128 KB
        # baseline has 61148 parameters -> we can afford 16-bit precision
        # since 61148 * 16 bit ~ 122 kB
        #obtain ckpt_id from config
        ckpt_id = self.config.ckpt_id
        
        # Determine the CAM method and append it to the folder name
        cam_method_name = self.config.CamMethod  # Ensure this attribute is set with the method name (e.g., "HiResCAM")
        class_label = self.label_ids[labels.item()]# Directory to save activation maps
        ckpt_dir = os.path.join("checkpoints", ckpt_id)
        os.makedirs(ckpt_dir, exist_ok=True)
        save_path = os.path.join(ckpt_dir, f"{cam_method_name}_activation_maps.pt")
        
        with torch.set_grad_enabled(True):
            if self.config.Cam:
                with torch.enable_grad():
                    x = self.mel_forward(x)
                    x.requires_grad_()  # Enable gradients for GradCAM
                    # print(f"{x.shape}\n")
                    # Define target layer and initialize selected CAM method
                    target_layer = self.model.blocks[-1].norm1
                    # print(f"Obtained target layer: {target_layer} \n")
                    if cam_method_name == "HiResCAM":
                        cam = HiResCAM(model=self.model, target_layers=[target_layer], use_cuda=True) #reshape_transform=self.reshape_transform,
                    elif cam_method_name == "GradCAM":
                        # print(f"getting GradCAM... \n")
                        cam = GradCAM(model=self.model, target_layers=[target_layer],reshape_transform=self.reshape_transform, use_cuda=True)
                        # print(f"Passed getting GradCam \n")
                    elif cam_method_name == "GradCAMPlusPlus":
                        cam = GradCAMPlusPlus(model=self.model, target_layers=[target_layer], use_cuda=True)
                    elif cam_method_name == 'DeepFeatureFactorization':
                        cam = DeepFeatureFactorization(model=self.model, target_layer=[target_layer])
                    else:
                        raise ValueError(f"Unsupported CAM method: {cam_method_name}")
                    y_hat, embed = self.forward(x)
                    targets = [ClassifierOutputTarget(labels.item())]
                    
                    # Generate GradCAM activation maps
                    # print("Generating GradCAM activation maps...")
                    activation_maps = cam(input_tensor=x, targets=targets, aug_smooth=self.aug_smooth)
                    # print("Activation maps generated successfully")
                    
                    # Accumulate data for the epoch
                    self.epoch_activation_maps.append({
                    "activation_maps": activation_maps,
                    "file": files[0],
                    "label": class_label
                })
                return None
            else:
                x = self.mel_forward(x)
                y_hat, embed = self.forward(x)
                labels = labels.long()
                samples_loss = F.cross_entropy(y_hat, labels, reduction="none")
                # loss = samples_loss.mean()

                # for computing accuracy
                _, preds = torch.max(y_hat, dim=1)
                n_correct_per_sample = (preds == labels)
                n_correct = n_correct_per_sample.sum()

                dev_names = [d.rsplit("-", 1)[1][:-4] for d in files]
                results = {'loss': samples_loss.mean(), "n_correct": n_correct,
                        "n_pred": torch.as_tensor(len(labels), device=self.device)}

                # log metric per device and scene
                for d in self.device_ids:
                    results["devloss." + d] = torch.as_tensor(0., device=self.device)
                    results["devcnt." + d] = torch.as_tensor(0., device=self.device)
                    results["devn_correct." + d] = torch.as_tensor(0., device=self.device)
                for i, d in enumerate(dev_names):
                    results["devloss." + d] = results["devloss." + d] + samples_loss[i]
                    results["devn_correct." + d] = results["devn_correct." + d] + n_correct_per_sample[i]
                    results["devcnt." + d] = results["devcnt." + d] + 1

                for l in self.label_ids:
                    results["lblloss." + l] = torch.as_tensor(0., device=self.device)
                    results["lblcnt." + l] = torch.as_tensor(0., device=self.device)
                    results["lbln_correct." + l] = torch.as_tensor(0., device=self.device)
                for i, l in enumerate(labels):
                    results["lblloss." + self.label_ids[l]] = results["lblloss." + self.label_ids[l]] + samples_loss[i]
                    results["lbln_correct." + self.label_ids[l]] = \
                        results["lbln_correct." + self.label_ids[l]] + n_correct_per_sample[i]
                    results["lblcnt." + self.label_ids[l]] = results["lblcnt." + self.label_ids[l]] + 1
                self.test_step_outputs.append(results)

    def on_test_epoch_end(self):
        if self.config.Cam and self.epoch_activation_maps:
            ckpt_id = self.config.ckpt_id
            cam_method_name = self.config.CamMethod
            target_class = self.config.target_class  # Retrieve the target class
            label_list = {label: idx for idx, label in enumerate(self.label_ids)}
            class_idx = label_list[target_class]
            class_label = self.label_ids[class_idx]
            out_dir = os.path.join("cams", ckpt_id+'_'+ cam_method_name)
            os.makedirs(out_dir, exist_ok=True)
            save_path = os.path.join(out_dir, f"{class_label}_activation_maps.pt")
            out_dir = os.path.join("cams", ckpt_id+'_'+ cam_method_name)

            # Save all activation maps as a single .pt file
            torch.save({
                "activation_maps": [item["activation_maps"] for item in self.epoch_activation_maps],
                "files": [item["file"] for item in self.epoch_activation_maps],
                "labels": [item["label"] for item in self.epoch_activation_maps],
                "cam_method": cam_method_name
            }, save_path)
            print(f"All activation maps saved at {save_path}")

            # Clear storage for next epoch
            self.epoch_activation_maps = []
        else:
            # convert a list of dicts to a flattened dict
            outputs = {k: [] for k in self.test_step_outputs[0]}
            for step_output in self.test_step_outputs:
                for k in step_output:
                    outputs[k].append(step_output[k])
            for k in outputs:
                outputs[k] = torch.stack(outputs[k])

            avg_loss = outputs['loss'].mean()
            acc = sum(outputs['n_correct']) * 1.0 / sum(outputs['n_pred'])

            logs = {'acc': acc, 'loss': avg_loss}

            # log metric per device and scene
            for d in self.device_ids:
                dev_loss = outputs["devloss." + d].sum()
                dev_cnt = outputs["devcnt." + d].sum()
                dev_corrct = outputs["devn_correct." + d].sum()
                logs["loss." + d] = dev_loss / dev_cnt
                logs["acc." + d] = dev_corrct / dev_cnt
                logs["cnt." + d] = dev_cnt
                # device groups
                logs["acc." + self.device_groups[d]] = logs.get("acc." + self.device_groups[d], 0.) + dev_corrct
                logs["count." + self.device_groups[d]] = logs.get("count." + self.device_groups[d], 0.) + dev_cnt
                logs["lloss." + self.device_groups[d]] = logs.get("lloss." + self.device_groups[d], 0.) + dev_loss

            for d in set(self.device_groups.values()):
                logs["acc." + d] = logs["acc." + d] / logs["count." + d]
                logs["lloss." + d] = logs["lloss." + d] / logs["count." + d]

            for l in self.label_ids:
                lbl_loss = outputs["lblloss." + l].sum()
                lbl_cnt = outputs["lblcnt." + l].sum()
                lbl_corrct = outputs["lbln_correct." + l].sum()
                logs["loss." + l] = lbl_loss / lbl_cnt
                logs["acc." + l] = lbl_corrct / lbl_cnt
                logs["cnt." + l] = lbl_cnt

            logs["macro_avg_acc"] = torch.mean(torch.stack([logs["acc." + l] for l in self.label_ids]))
            # prefix with 'test' for logging
            self.log_dict({"test/" + k: logs[k] for k in logs})
            self.test_step_outputs.clear()
        
    def configure_optimizers(self):
        """
        This is the way pytorch lightening requires optimizers and learning rate schedulers to be defined.
        The specified items are used automatically in the optimization loop (no need to call optimizer.step() yourself).
        :return: dict containing optimizer and learning rate scheduler
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        schedule_lambda = \
            exp_warmup_linear_down(self.config.warm_up_len, self.config.ramp_down_len, self.config.ramp_down_start,
                                   self.config.last_lr_value)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_lambda)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler
        }
    
def evaluate(config):
    import os
    from sklearn import preprocessing
    import pandas as pd
    import torch.nn.functional as F
    from datasets.dcase23_dev import dataset_config

    assert config.ckpt_id is not None, "A value for argument 'ckpt_id' must be provided."
    ckpt_dir = os.path.join(config.project_name, config.ckpt_id, "checkpoints")
    assert os.path.exists(ckpt_dir), f"No such folder: {ckpt_dir}"
    #ckpt_file = os.path.join(ckpt_dir, "last.ckpt")
    for file in os.listdir(ckpt_dir):
        if "epoch" in file:
            ckpt_file = os.path.join(ckpt_dir,file) # choosing the best model ckpt
            print(f"found ckpt file: {file}")
    # ckpt_file = os.path.join(ckpt_dir, "xyz.ckpt") # Change the path to the model path desired
    assert os.path.exists(ckpt_file), f"No such file: {ckpt_file}. Implement your own mechanism to select" \
                                      f"the desired checkpoint."

    # create folder to store predictions
    os.makedirs("predictions", exist_ok=True)
    out_dir = os.path.join("predictions", config.ckpt_id)
    os.makedirs(out_dir, exist_ok=True)

    # load lightning module from checkpoint
    # pl_module = PLModule.load_from_checkpoint(ckpt_file, config=config)
    
    
    # load lightning module from checkpoint
    pl_module = PLModule.load_from_checkpoint(ckpt_file, config=config)
    pl_module.config.Cam = config.Cam
    eval_ds = get_test_set()
    if config.Cam and config.target_class is not None:
        # Load evaluation dataset and select the single sample specified by CAM_index
        target_class = config.target_class
        test_csv = os.path.join(dataset_config['split_path'], dataset_config['test_split_csv'])
        data = pd.read_csv(test_csv, sep="\t")
        indices = data.index[data["scene_label"] == target_class].tolist()
        print(f"Selected {len(eval_ds)} samples for class {target_class}")
        eval_ds = Subset(eval_ds,indices)
        # Configure the DataLoader
        test_dl = DataLoader(
            dataset=eval_ds,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            worker_init_fn=worker_init_fn,
            shuffle=False
        )

        # Initialize the PyTorch Lightning Trainer and run testing
        trainer = pl.Trainer(
            logger=False,
            accelerator="gpu",
            devices=[0],
            precision=config.precision,inference_mode=False
        )
        trainer.test(pl_module, test_dl)
    else:
        trainer = pl.Trainer(logger=False,
                            accelerator='gpu',
                            devices=1,
                            precision=config.precision)
        ############# h5 edit here ##############
        # evaluate lightning module on development-test split
        test_dl = DataLoader(dataset=get_test_set(),
                            worker_init_fn=worker_init_fn,
                            num_workers=config.num_workers,
                            batch_size=config.batch_size,
                            pin_memory=True)

        # get model complexity from nessi
        sample = next(iter(test_dl))[0][0].unsqueeze(0).to(pl_module.device)
        shape = pl_module.mel_forward(sample).size()
        macs, params = nessi.get_torch_size(pl_module.model, input_size=shape)

        print(f"Model Complexity: MACs: {macs}, Params: {params}")
        # assert macs <= nessi.MAX_MACS, "The model exceeds the MACs limit and must not be submitted to the challenge!"
        # assert params <= nessi.MAX_PARAMS_MEMORY, \
            # "The model exceeds the parameter limit and must not be submitted to the challenge!"

        allowed_precision = int(nessi.MAX_PARAMS_MEMORY / params * 8)
        print(f"ATTENTION: According to the number of model parameters and the memory limits that apply in the challenge,"
            f" you are allowed to use at max the following precision for model parameters: {allowed_precision} bit.")

        # obtain and store details on model for reporting in the technical report
        info = {}
        info['MACs'] = macs
        info['Params'] = params
        res = trainer.test(pl_module, test_dl,ckpt_path=ckpt_file)
        info['test'] = res

        ############# h5 edit here ##############
        # generate predictions on evaluation set
        eval_dl = DataLoader(dataset=get_eval_set(),
                            worker_init_fn=worker_init_fn,
                            num_workers=config.num_workers,
                            batch_size=config.batch_size)
        
        predictions = trainer.predict(pl_module, dataloaders=eval_dl, ckpt_path=ckpt_file) # predictions returns as files, y_hat
        # all filenames
        all_files = [item[len("audio/"):] for files, _ in predictions for item in files]
        # all predictions
        logits = torch.cat([torch.as_tensor(p) for _, p in predictions], 0)
        all_predictions = F.softmax(logits.float(), dim=1)

        # write eval set predictions to csv file
        df = pd.read_csv(dataset_config['meta_csv'], sep="\t")
        le = preprocessing.LabelEncoder()
        le.fit_transform(df[['scene_label']].values.reshape(-1))
        class_names = le.classes_
        df = {'filename': all_files}
        scene_labels = [class_names[i] for i in torch.argmax(all_predictions, dim=1)]
        df['scene_label'] = scene_labels
        for i, label in enumerate(class_names):
            df[label] = logits[:, i]
        df = pd.DataFrame(df)

        # save eval set predictions, model state_dict and info to output folder
        df.to_csv(os.path.join(out_dir, 'output.csv'), sep='\t', index=False)
        torch.save(pl_module.model.state_dict(), os.path.join(out_dir, "model_state_dict.pt"))
        with open(os.path.join(out_dir, "info.json"), "w") as json_file:
            json.dump(info, json_file)
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')

    # general
    parser.add_argument('--project_name', type=str, default="NTU24_ASC")
    parser.add_argument('--experiment_name', type=str, default="NTU_passt_FTtau_441K_FMS_fixh5")
    parser.add_argument('--num_workers', type=int, default=0)  # number of workers for dataloaders
    parser.add_argument('--precision', type=str, default="32")
    

    # evaluation
    parser.add_argument('--evaluate', action='store_true',default=True)  # predictions on eval set
    parser.add_argument('--ckpt_id', type=str, default="jiw5bohu")  # for loading trained model, corresponds to wandb id
    
    # GRADCAM
    parser.add_argument('--Cam', action='store_true', default=True, help='Use GradCAM for visualizing class activations')
    # parser.add_argument('--Cam_index', type=int, default=14840, help='Index of the audio sample to visualize with GradCAM')
    # parser.add_argument('--Cam_index_range', type=int, nargs=2, default=[2960,5930], help="Range of indices for evaluation dataset (start end).")
    parser.add_argument('--target_class', type=str, default="airport", help="Select class to visualize and compare.")
    parser.add_argument('--CamMethod', type=str,default='GradCAM', help='Choose from GradCAM, HiResCAM, GradCAMPlusPlus, DeepFeatureFactorization') 
    parser.add_argument('--aug_smooth',action='store_true', default= False, help= 'toggle smoothing')
    
    # dataset
    # location to store resampled waveform
    parser.add_argument('--cache_path', type=str, default=os.path.join("datasets", "cpath"))
    parser.add_argument('--subset', type=int, default=5)
    # model
    parser.add_argument('--arch', type=str, default='passt_s_swa_p16_128_ap476')  # pretrained passt model
    parser.add_argument('--n_classes', type=int, default=10)  # classification model with 'n_classes' output neurons
    parser.add_argument('--input_fdim', type=int, default=128)
    parser.add_argument('--s_patchout_t', type=int, default=0)
    parser.add_argument('--s_patchout_f', type=int, default=6)

    # training
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--mixstyle_p', type=float, default=0.4)  # frequency mixstyle
    parser.add_argument('--mixstyle_alpha', type=float, default=0.4)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--roll', type=int, default=4000)  # roll waveform over time
    parser.add_argument('--dir_prob', type=float, default=0 ) # prob. to apply device impulse response augmentation # need to specify
    # parser.add_argument('--mixup_alpha', type=float, default=1.0)
    # learning rate + schedule
    # phases:
    #  1. exponentially increasing warmup phase (for 'warm_up_len' epochs)
    #  2. constant lr phase using value specified in 'lr' (for 'ramp_down_start' - 'warm_up_len' epochs)
    #  3. linearly decreasing to value 'las_lr_value' * 'lr' (for 'ramp_down_len' epochs)
    #  4. finetuning phase using a learning rate of 'last_lr_value' * 'lr' (for the rest of epochs up to 'n_epochs')
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--warm_up_len', type=int, default=3)
    parser.add_argument('--ramp_down_start', type=int, default=3)
    parser.add_argument('--ramp_down_len', type=int, default=10)
    parser.add_argument('--last_lr_value', type=float, default=0.01)  # relative to 'lr'

    # preprocessing
    parser.add_argument('--resample_rate', type=int, default=44100) # default =32000
    parser.add_argument('--window_size', type=int, default=800)  # in samples
    parser.add_argument('--hop_size', type=int, default=320)  # in samples
    parser.add_argument('--n_fft', type=int, default=1024)  # length (points) of fft
    parser.add_argument('--n_mels', type=int, default=128)  # number of mel bins
    parser.add_argument('--freqm', type=int, default=48)  # mask up to 'freqm' spectrogram bins
    parser.add_argument('--timem', type=int, default=20)  # mask up to 'timem' spectrogram frames # can try 192
    parser.add_argument('--fmin', type=int, default=0)  # mel bins are created for freqs. between 'fmin' and 'fmax'
    parser.add_argument('--fmax', type=int, default=None)
    parser.add_argument('--fmin_aug_range', type=int, default=1)  # data augmentation: vary 'fmin' and 'fmax'
    parser.add_argument('--fmax_aug_range', type=int, default=1000)

    args = parser.parse_args()
    evaluate(args)
    