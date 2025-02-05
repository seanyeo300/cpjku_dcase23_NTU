import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import argparse
import os
import torchaudio
import wandb
import optuna

from torch.autograd import Variable
from helpers.lr_schedule import exp_warmup_linear_down
from helpers.init import worker_init_fn
from models.passt import get_model
from models.mel import AugmentMelSTFT
from helpers import nessi
# from datasets.dcase23_dev import get_training_set, get_test_set, get_eval_set
from datasets.cochl_ntu_teacher_mel import ntu_get_training_set_dir, ntu_get_test_set, ntu_get_test_set_cochl, ntu_get_eval_set, open_h5, close_h5,ntu_cochl_subset_dir, ntu_cochl_val_set, ntu_get_test_subset_cochl
from helpers.utils import mixstyle, mixup_data
import json

torch.set_float32_matmul_precision("high")
def load_and_modify_checkpoint(pl_module, num_classes=10):
    # Modify the final layer
    pl_module.model.head = nn.Sequential(
        nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True),
        nn.Linear(768, num_classes)
    )
    return pl_module

def objective(trial):
    # Suggest temperature and KD_lambda hyperparameters for this trial
    # temperature = trial.suggest_float('temperature', 1.0,4.0)  # temperature scaling for KD
    lr = trial.suggest_float('lr', 1e-6, 1e-4)  # FG learning rate range

    # Update the config with Optuna-suggested hyperparameters
    config = argparse.Namespace(
        project_name="Optuna_KD",
        experiment_name="Optuna_SIT_FG_lr_Trial_" + str(trial.number),
        num_workers=0,
        gpu="[0]",
        precision="32",
        evaluate=False,
        ckpt_id=None,
        orig_sample_rate=44100,
        subset="cochlOLsub100",
        n_classes=5,
        arch='passt_s_swa_p16_128_ap476',
        input_fdim=128,
        s_patchout_f=6,
        s_patchout_t=0,
        n_epochs=25,
        batch_size=32,
        mixstyle_p=0.4,
        mixstyle_alpha=0.4,
        weight_decay=0.001,
        roll_sec=4000,
        dir_prob=0,
        lr=lr,
        warm_up_len=3,
        ramp_down_start=3,
        ramp_down_len=10,
        last_lr_value=0.01,
        resample_rate=44100,
        window_size=800,
        hop_size=320,
        n_fft=1024,
        n_mels=128,
        freqm=48,
        timem=20,
        fmin=0,
        fmax=None,
        fmin_aug_range=1,
        fmax_aug_range=1000
    )

    # Initialize WandB logger with trial-specific settings
    wandb_logger = WandbLogger(
        project=config.project_name,
        notes="Optimization study for KD",
        tags=["DCASE24"],
        config=config,
        name=config.experiment_name
    )

    # Set up DataLoader and model as in your training function
    hf_in = open_h5('h5py_cochl_train_mel_10s_128bins')
    hmic_in = open_h5('h5py_mic_wav_1')
    
    train_dl = DataLoader(dataset=ntu_cochl_subset_dir(config.subset, config.dir_prob, hf_in, hmic_in),
                          worker_init_fn=worker_init_fn,
                          num_workers=config.num_workers,
                          batch_size=config.batch_size,
                          
                          shuffle=True)
    
    val_dl = DataLoader(dataset=ntu_cochl_val_set(hf_in),
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size,
                         pin_memory=True,
                         )
    test_dl = DataLoader(dataset=ntu_get_test_subset_cochl(hf_in),
                         worker_init_fn=worker_init_fn,
                         num_workers=config.num_workers,
                         batch_size=config.batch_size,
                         pin_memory=True,
                         )

    
    pl_module = PLModule(config)

    # Log model complexity (optional)
    sample = next(iter(test_dl))[0][0].unsqueeze(0)
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
    # Trainer setup
    trainer = pl.Trainer(max_epochs=config.n_epochs,
                         logger=wandb_logger,
                         accelerator='gpu',
                         devices=eval(config.gpu),
                         callbacks=[lr_monitor, checkpoint_callback])

    # Train and validate
    trainer.fit(pl_module, train_dl, val_dl)
    # Test and retrieve metric for Optuna to optimize
    results = trainer.test(ckpt_path='best', dataloaders=test_dl)
    
    accuracy = results[0]['test/macro_avg_acc']  # Specify the exact metric key here
    # val_loss = trainer.callback_metrics['val/loss'].item()  # Retrieve the validation loss
    
    # Clean up
    wandb.finish()
    close_h5(hf_in)
    close_h5(hmic_in)
    return accuracy

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

        # self.device_ids = ['a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']
        # self.label_ids = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square', 'shopping_mall',
        #                   'street_pedestrian', 'street_traffic', 'tram']
        self.label_ids = ['Bus', 'Park', 'Street','SubwayStation', 'Subway']
        # ['Bus', 'Cafe', 'Car', 'CrowdedIndoor', 'Elevator', 'Kitchen','Park', 'ResidentialArea', 'Restaurant', 'Restroom', 'Street','SubwayStation', 'Subway']
        # 
        # ['Bus', 'Park', 'Street','SubwayStation', 'Subway']
        self.device_groups = {'a': "real", 'b': "real", 'c': "real",
                              's1': "seen", 's2': "seen", 's3': "seen",
                              's4': "unseen", 's5': "unseen", 's6': "unseen"}

        self.calc_device_info = True
        self.epoch = 0
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        freqm = torchaudio.transforms.FrequencyMasking(config.freqm, iid_masks=True)
        timem = torchaudio.transforms.TimeMasking(config.timem, iid_masks=True)
        self.mel_augment = torch.nn.Sequential(
            freqm,
            timem
        )
        
    def mel_forward(self, x): ###### modified for use with mel spec precomputed #######
        # old_shape = x.size()
        # x = x.reshape(-1, old_shape[2])
        # x = self.mel(x)
        # x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
        if self.training:
            x = self.mel_augment(x)
            
        return x

    def forward(self, x):
        return self.model(x)
    
    def predict_step(self, eval_batch, batch_idx, dataloader_idx=0):
        x, files = eval_batch
        x = self.mel_forward(x)
        # x = x.half()
        y_hat, embed = self.forward(x)

        return files, y_hat
    # def mixup_criterion(self,criterion, pred, y_a, y_b, lam):
    #         return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
        
    def training_step(self, batch, batch_idx):
        criterion = torch.nn.CrossEntropyLoss()
        # x, files, labels, devices, cities, teacher_logits = batch
        x, files, labels, devices, cities = batch

        if self.mel:
            x = self.mel_forward(x)

        if self.config.mixstyle_p > 0:
            x = mixstyle(x, self.config.mixstyle_p, self.config.mixstyle_alpha)
            
        y_hat, embed = self.forward(x)
        labels = labels.long()
        # inputs, targets_a, targets_b, lam = mixup_data(x, labels,
                                                    #    self.config.mixup_alpha, use_cuda=True)
        # inputs, targets_a, targets_b = map(Variable, (inputs,
                                                    #   targets_a, targets_b))
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")

        # loss = self.mixup_criterion(criterion,y_hat, targets_a, targets_b, lam)

        loss = samples_loss.mean()
        samples_loss = samples_loss.detach()

        _, preds = torch.max(y_hat, dim=1)
        n_correct_pred = (preds == labels).sum()
        results = {"loss": loss, "n_correct_pred": n_correct_pred, "n_pred": len(labels)}

        # if self.calc_device_info:
        #     devices = [d.rsplit("-", 1)[1][:-4] for d in files]

        #     for d in self.device_ids:
        #         results["devloss." + d] = torch.as_tensor(0., device=self.device)
        #         results["devcnt." + d] = torch.as_tensor(0., device=self.device)

        #     for i, d in enumerate(devices):
        #         results["devloss." + d] = results["devloss." + d] + samples_loss[i]
        #         results["devcnt." + d] = results["devcnt." + d] + 1.
        # print(f"Training Loss = {loss}")
        return results

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        train_acc = sum([x['n_correct_pred'] for x in outputs]) * 1.0 / sum(x['n_pred'] for x in outputs)
        logs = {'train.loss': avg_loss, 'train_acc': train_acc}

        # if self.calc_device_info:
        #     for d in self.device_ids:
        #         dev_loss = torch.stack([x["devloss." + d] for x in outputs]).sum()
        #         dev_cnt = torch.stack([x["devcnt." + d] for x in outputs]).sum()
        #         logs["tloss." + d] = dev_loss / dev_cnt
        #         logs["tcnt." + d] = dev_cnt

        self.log_dict(logs)

        print(f"Training Loss: {avg_loss}")
        print(f"Training Accuracy: {train_acc}")


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
           
        x = self.mel_forward(x)

        y_hat, embed = self.forward(x)
        labels = labels.long()
        samples_loss = F.cross_entropy(y_hat, labels, reduction="none")
        # loss = samples_loss.mean()

        # for computing accuracy
        _, preds = torch.max(y_hat, dim=1)
        n_correct_per_sample = (preds == labels)
        n_correct = n_correct_per_sample.sum()

        # dev_names = [d.rsplit("", 1)[1][:-4] for d in files]
        results = {'loss': samples_loss.mean(), "n_correct": n_correct,
                   "n_pred": torch.as_tensor(len(labels), device=self.device)}

        # log metric per device and scene
        # for d in self.device_ids:
        #     results["devloss." + d] = torch.as_tensor(0., device=self.device)
        #     results["devcnt." + d] = torch.as_tensor(0., device=self.device)
        #     results["devn_correct." + d] = torch.as_tensor(0., device=self.device)
        # for i, d in enumerate(dev_names):
        #     results["devloss." + d] = results["devloss." + d] + samples_loss[i]
        #     results["devn_correct." + d] = results["devn_correct." + d] + n_correct_per_sample[i]
        #     results["devcnt." + d] = results["devcnt." + d] + 1

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
        # for d in self.device_ids:
        #     dev_loss = outputs["devloss." + d].sum()
        #     dev_cnt = outputs["devcnt." + d].sum()
        #     dev_corrct = outputs["devn_correct." + d].sum()
        #     logs["loss." + d] = dev_loss / dev_cnt
        #     logs["acc." + d] = dev_corrct / dev_cnt
        #     logs["cnt." + d] = dev_cnt
        #     # device groups
        #     logs["acc." + self.device_groups[d]] = logs.get("acc." + self.device_groups[d], 0.) + dev_corrct
        #     logs["count." + self.device_groups[d]] = logs.get("count." + self.device_groups[d], 0.) + dev_cnt
        #     logs["lloss." + self.device_groups[d]] = logs.get("lloss." + self.device_groups[d], 0.) + dev_loss

        # for d in set(self.device_groups.values()):
        #     logs["acc." + d] = logs["acc." + d] / logs["count." + d]
        #     logs["lloss." + d] = logs["lloss." + d] / logs["count." + d]

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
        representation_params = []
        for name, param in self.model.named_parameters():
            if 'head' not in name and 'head_dist' not in name:
                representation_params.append(param)
        # Collect parameters for the classification heads
        head_params = list(self.model.head.parameters()) + list(self.model.head_dist.parameters())
        # Define the optimizer with parameter groups
        optimizer = torch.optim.Adam([
            {'params': representation_params, 'lr': self.config.lr},  # Low learning rate for representation layers
            {'params': head_params, 'lr': 0.001},             # High learning rate for classifier heads
        ], weight_decay=self.config.weight_decay)
        schedule_lambda = \
            exp_warmup_linear_down(self.config.warm_up_len, self.config.ramp_down_len, self.config.ramp_down_start,
                                   self.config.last_lr_value)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_lambda)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler
        }


    
if __name__ == '__main__':
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25)

    print("Best trial:")
    print("  Value:", study.best_trial.value)
    print("  Params:", study.best_trial.params)