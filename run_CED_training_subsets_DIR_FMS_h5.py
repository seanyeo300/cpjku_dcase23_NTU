import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import argparse
import os
import torch.nn as nn
import torchaudio
import transformers

from torch.autograd import Variable
from helpers.lr_schedule import exp_warmup_linear_down
from helpers.init import worker_init_fn
import models.CED.audiotransformer
from models.CED.checkpoints import list_models
from models.mel import AugmentMelSTFT
from helpers import nessi
# from datasets.dcase23_dev import get_training_set, get_test_set, get_eval_set
from datasets.dcase24_ntu_teacher import ntu_get_training_set_dir, ntu_get_test_set, ntu_get_eval_set, open_h5, close_h5
from helpers.utils import mixstyle, mixup_data
import json

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

        resample = torchaudio.transforms.Resample(
            orig_freq=self.config.orig_sample_rate,
            new_freq=self.config.resample_rate
        )
        self.config = config
        self.samp = torch.nn.Sequential(
            resample
        )
        # model to preprocess waveforms into log mel spectrograms
        # self.mel = AugmentMelSTFT(n_mels=config.n_mels,
        #                           sr=config.resample_rate,
        #                           win_length=config.window_size,
        #                           hopsize=config.hop_size,
        #                           n_fft=config.n_fft,
        #                           freqm=config.freqm,
        #                           timem=config.timem,
        #                           fmin=config.fmin,
        #                           fmax=config.fmax,
        #                           fmin_aug_range=config.fmin_aug_range,
        #                           fmax_aug_range=config.fmax_aug_range
        #                           )

        self.model = getattr(models.CED.audiotransformer, config.arch)(pretrained=True)

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
        
    def mel_forward(self, x):
        x = self.samp(x)
        return x

    def forward(self, x):
        return self.model(x)
    
    def predict_step(self, eval_batch, batch_idx, dataloader_idx=0):
        x, files = eval_batch
        # x = self.mel_forward(x)
        y_hat, embed = self.forward(x)

        return files, y_hat
    # def mixup_criterion(self,criterion, pred, y_a, y_b, lam):
    #         return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
        
    def training_step(self, batch, batch_idx):
        criterion = torch.nn.CrossEntropyLoss()
        x, files, labels, devices, cities, teacher_logits = batch
        
        x = self.mel_forward(x) # resamples the audio (FOR CED only)
        for chunk_idx, chunk in enumerate(
                x.split(int(self.config.chunk_length * 16000), -1)):
            logits = self.forward(chunk).squeeze(0)  # Get model outputs (logits)
            probabilities = F.softmax(logits, dim=0)      # Apply softmax to get probabilities
            print(f"probabilities from CED model: {probabilities}")
            print(f"logits look like this: {logits}")
        # y_hat, embed = self.forward(x)
        labels = labels.long()
        label_maps = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square', 'shopping_mall','street_pedestrian', 'street_traffic', 'tram']
        pred_idx = torch.argmax(probabilities).item()
        print(f"final prediction index looks like this: {pred_idx}")
        # inputs, targets_a, targets_b, lam = mixup_data(x, labels,
                                                    #    self.config.mixup_alpha, use_cuda=True)
        # inputs, targets_a, targets_b = map(Variable, (inputs,
                                                    #   targets_a, targets_b))
        y_hat=logits
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")

        # loss = self.mixup_criterion(criterion,y_hat, targets_a, targets_b, lam)

        loss = samples_loss.mean()
        samples_loss = samples_loss.detach()

        # _, preds = torch.max(y_hat, dim=1)
        preds=pred_idx
        n_correct_pred = (preds == labels).sum()
        results = {"loss": loss, "n_correct_pred": n_correct_pred, "n_pred": len(labels)}

        if self.calc_device_info:
            devices = [d.rsplit("-", 1)[1][:-4] for d in files]

            for d in self.device_ids:
                results["devloss." + d] = torch.as_tensor(0., device=self.device)
                results["devcnt." + d] = torch.as_tensor(0., device=self.device)

            for i, d in enumerate(devices):
                results["devloss." + d] = results["devloss." + d] + samples_loss[i]
                results["devcnt." + d] = results["devcnt." + d] + 1.

        return results

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        train_acc = sum([x['n_correct_pred'] for x in outputs]) * 1.0 / sum(x['n_pred'] for x in outputs)
        logs = {'train.loss': avg_loss, 'train_acc': train_acc}

        if self.calc_device_info:
            for d in self.device_ids:
                dev_loss = torch.stack([x["devloss." + d] for x in outputs]).sum()
                dev_cnt = torch.stack([x["devcnt." + d] for x in outputs]).sum()
                logs["tloss." + d] = dev_loss / dev_cnt
                logs["tcnt." + d] = dev_cnt

        self.log_dict(logs)

        print(f"Training Loss: {avg_loss}")
        print(f"Training Accuracy: {train_acc}")


    def validation_step(self, batch, batch_idx):
        x, files, labels, devices, cities = batch
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )


def train(config):
    # logging is done using wandb
    wandb_logger = WandbLogger(
        project=config.project_name,
        notes="CED Training for DCASE2024, incl subset rules.",
        tags=["DCASE24"],
        config=config,  # this logs all hyperparameters for us
        name=config.experiment_name
    )

    ######### h5 edit here ###############
    # get pointer to h5 file containing audio samples
    hf_in = open_h5('h5py_audio_wav')
    hmic_in = open_h5('h5py_mic_wav_1')

    # get_training set already as logic to handle dir_prob=0
    train_dl = DataLoader(dataset=ntu_get_training_set_dir(config.subset, config.dir_prob, hf_in, hmic_in),
                          worker_init_fn=worker_init_fn,
                          num_workers=config.num_workers,
                          batch_size=config.batch_size,
                          
                          shuffle=True)
    
    test_dl = DataLoader(dataset=ntu_get_test_set(hf_in),
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size)

    # create pytorch lightening module
    pl_module = PLModule(config) # this initializes the model pre-trained on audioset
    # ckpt_dir = os.path.join(config.project_name, config.ckpt_id, "checkpoints")
    # assert os.path.exists(ckpt_dir), f"No such folder: {ckpt_dir}"
    # #ckpt_file = os.path.join(ckpt_dir, "last.ckpt")
    # for file in os.listdir(ckpt_dir):
    #     if "epoch" in file:
    #         ckpt_file = os.path.join(ckpt_dir,file) # choosing the best model ckpt
    #         print(f"found ckpt file: {file}")
    # pl_module = PLModule.load_from_checkpoint(ckpt_file, config=config)
    # pl_module = load_and_modify_checkpoint(pl_module)
    # get model complexity from nessi and log results to wandb
    # ATTENTION: this is before layer fusion, therefore the MACs and Params slightly deviate from what is
    # reported in the challenge submission
    sample = next(iter(train_dl))[0][0].unsqueeze(0)
    shape = pl_module.mel_forward(sample).size()
    macs, params = nessi.get_torch_size(pl_module.model, input_size=shape)
    wandb_logger.experiment.config['MACs'] = macs
    wandb_logger.experiment.config['Parameters'] = params

    # create monitor to keep track of learning rate - we want to check the behaviour of our learning rate schedule
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_last=True, 
        monitor="validation.loss", 
        save_top_k=1)
    # create the pytorch lightening trainer by specifying the number of epochs to train, the logger,
    # on which kind of device(s) to train and possible callbacks
    trainer = pl.Trainer(max_epochs=config.n_epochs,
                         logger=wandb_logger,
                         accelerator='gpu',
                         devices=1,
                         callbacks=[lr_monitor, checkpoint_callback])
    # start training and validation for the specified number of epochs
    trainer.fit(pl_module, train_dl, test_dl)
    trainer.test(ckpt_path='best', dataloaders=test_dl)
    ############ h5 edit end #################
    # close file pointer to h5 file 
    close_h5(hf_in)
    close_h5(hmic_in)
    
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
    
    ############# h5 edit here ##############
    # Open h5 file once
    hf_in = open_h5('h5py_audio_wav')
    # eval_hf = open_h5('h5py_audio_wav2') # only when obtaining pre-computed train
    # eval_hf = open_h5('h5py_audio_eval_wav')
    # load lightning module from checkpoint
    pl_module = PLModule.load_from_checkpoint(ckpt_file, config=config)
    trainer = pl.Trainer(logger=False,
                         accelerator='gpu',
                         devices=1,
                         precision=config.precision)
    ############# h5 edit here ##############
    # evaluate lightning module on development-test split
    test_dl = DataLoader(dataset=ntu_get_test_set(hf_in),
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
    eval_dl = DataLoader(dataset=ntu_get_eval_set(hf_in),
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
        
    ############# h5 edit here ##############
    close_h5(hf_in)
    # close_h5(eval_hf)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')

    # general
    parser.add_argument('--project_name', type=str, default="NTU_h5_test")
    parser.add_argument('--experiment_name', type=str, default="NTU_PaSST_h5_441K_FMS_DIR_compare_outputs")
    parser.add_argument('--num_workers', type=int, default=0)  # number of workers for dataloaders
    parser.add_argument('--precision', type=str, default="32")
    
    # evaluation
    parser.add_argument('--evaluate', action='store_true')  # predictions on eval set
    parser.add_argument('--ckpt_id', type=str, default=None)  # for loading trained model, corresponds to wandb id

    # dataset
    # location to store resampled waveform
    parser.add_argument('--cache_path', type=str, default=os.path.join("datasets", "cpath"))
    parser.add_argument('--subset', type=int, default=5)
    # model
    parser.add_argument('--arch', type=str, default='ced_mini')  # pretrained passt model
    parser.add_argument('--n_classes', type=int, default=10)  # classification model with 'n_classes' output neurons
    # parser.add_argument('--input_fdim', type=int, default=128)
    # parser.add_argument('--s_patchout_t', type=int, default=0)
    # parser.add_argument('--s_patchout_f', type=int, default=6)

    # training
    parser.add_argument('--n_epochs', type=int, default=80) # Following DyMN, default=80
    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--dir_prob', type=float, default=0)  # prob. to apply device impulse response augmentation # need to specify
    # parser.add_argument('--mixup_alpha', type=float, default=1.0)
    
    # peak learning rate (in cosinge schedule)
    parser.add_argument('--lr', type=float, default=1e-4) # DyMN uses 1e-4
    parser.add_argument('--warmup_steps', type=int, default=100) # default = 2000, divide by 20 for 5% subset, 10 for 10%, 4 for 25%, 2 for 50%

    # preprocessing
    parser.add_argument('--resample_rate', type=int, default=16000) # default =32000
    parser.add_argument('--chunk_length', type=float, default=1.0) # Define audio sample length in seconds
    # parser.add_argument('--window_size', type=int, default=800)  # in samples
    # parser.add_argument('--hop_size', type=int, default=320)  # in samples
    # parser.add_argument('--n_fft', type=int, default=1024)  # length (points) of fft
    # parser.add_argument('--n_mels', type=int, default=128)  # number of mel bins
    # parser.add_argument('--freqm', type=int, default=48)  # mask up to 'freqm' spectrogram bins
    # parser.add_argument('--timem', type=int, default=20)  # mask up to 'timem' spectrogram frames # can try 192
    # parser.add_argument('--fmin', type=int, default=0)  # mel bins are created for freqs. between 'fmin' and 'fmax'
    # parser.add_argument('--fmax', type=int, default=None)
    # parser.add_argument('--fmin_aug_range', type=int, default=1)  # data augmentation: vary 'fmin' and 'fmax'
    # parser.add_argument('--fmax_aug_range', type=int, default=1000)

    args = parser.parse_args()
    if args.evaluate:
        evaluate(args)
    else:
        train(args)