import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint#, TQDMProgressBar
from torchsampler import ImbalancedDatasetSampler
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from scipy import stats
from SleepDataset import SleepSegmentationDataset
import argparse
import os
import warnings
from utils.load_config import load_config_as_dict
from utils.ts_reshape import one_hot_encode, repeat_labels
from models.unet import UNet1D

warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class NeuroTech_Trainer(pl.LightningModule):
    def __init__(self, args):
        super(NeuroTech_Trainer, self).__init__()
        self.hparams.update(args)
        self.model = UNet1D()
        self.save_hyperparameters()
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x.half())

    def training_step(self, batch, batch_idx):
        x, y = batch
        batch_size = x.size(0)
        y = y.type(torch.int64)
        downsample_ratio = self.hparams.downsample_ratio
        interval_len = 3000 // downsample_ratio
        try:
            y_mask = one_hot_encode(repeat_labels(y, interval_len)).cuda()
        except:
            print(torch.max(y))
            print(torch.min(y))
        # print(x.shape)
        x = x.permute(0, 2, 1, 3).reshape(batch_size, 6, -1)[:,:,::downsample_ratio]
        if self.hparams.input_norm:
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True)
            x = (x - mean) / (std + 1e-6)

        # print(x.shape, end='\n\n\n\n\n')
        outputs = self.forward(x.cuda())
        # print(outputs.shape, y_mask.shape) # torch.Size([1, 6, 84120]) torch.Size([1, 84120, 6])
        outputs = outputs.permute(0, 2, 1)
        loss = self.criterion(outputs, y_mask)
        
        # Step 1: Argmax across channels to get most likely class
        predicted_classes = torch.argmax(outputs, dim=2)  # outputs Shape (B, seq_len, 6)

        # Step 2: Majority vote (mode) for every 30 indices
        B, seq_len = predicted_classes.shape
        reduced_seq_len = seq_len // interval_len
        preds = torch.zeros((B, reduced_seq_len), dtype=predicted_classes.dtype)
        # print(preds.shape, y.shape, end='\n\n\n\n\n')
        for i in range(reduced_seq_len):
            sliced = predicted_classes[:, i*interval_len:(i+1)*interval_len]
            mode_values, _ = stats.mode(sliced.cpu(), axis=1)
            preds[:, i] = torch.from_numpy(mode_values).squeeze()
        
        # Step 3: Calculate accuracy
        acc = (preds.cuda() == y).sum().float() / y.numel()
        
        self.log('train_batch_acc', acc)
        self.log('train_batch_loss', loss)
        return {'loss': loss, 'train_acc': acc, 'preds':preds, 'y':y}

    def training_epoch_end(self, outputs):
        avg_loss = torch.tensor([x['loss'] for x in outputs]).mean()
        self.log('train_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=False)
        avg_acc = torch.stack([x['train_acc'] for x in outputs]).mean()
        self.log('train_acc', avg_acc, on_step=False, on_epoch=True, prog_bar=False)    

    def validation_step(self, batch, batch_idx):        
        x, y = batch
        batch_size = x.size(0)
        y = y.type(torch.int64)
        downsample_ratio = self.hparams.downsample_ratio
        interval_len = 3000 // downsample_ratio
        y_mask = one_hot_encode(repeat_labels(y, interval_len)).cuda()
        # print(x.shape)
        x = x.permute(0, 2, 1, 3).reshape(batch_size, 6, -1)[:,:,::downsample_ratio]
        if self.hparams.input_norm:
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True)
            x = (x - mean) / (std + 1e-6)

        # print(x.shape, end='\n\n\n\n\n')
        outputs = self.forward(x.cuda())
        # print(outputs.shape, y_mask.shape) # torch.Size([1, 6, 84120]) torch.Size([1, 84120, 6])
        outputs = outputs.permute(0, 2, 1)
        loss = self.criterion(outputs, y_mask)
        
        # Step 1: Argmax across channels to get most likely class
        predicted_classes = torch.argmax(outputs, dim=2)  # outputs Shape (B, seq_len, 6)

        # Step 2: Majority vote (mode) for every 30 indices
        B, seq_len = predicted_classes.shape
        reduced_seq_len = seq_len // interval_len
        preds = torch.zeros((B, reduced_seq_len), dtype=predicted_classes.dtype)
        # print(preds.shape, y.shape, end='\n\n\n\n\n')
        for i in range(reduced_seq_len):
            sliced = predicted_classes[:, i*interval_len:(i+1)*interval_len]
            mode_values, _ = stats.mode(sliced.cpu(), axis=1)
            preds[:, i] = torch.from_numpy(mode_values).squeeze()
        
        # Step 3: Calculate accuracy
        acc = (preds.cuda() == y).sum().float() / y.numel()
        
        self.log('train_batch_acc', acc)
        self.log('train_batch_loss', loss)
        return {'val_loss': loss, 'val_acc': acc, 'preds':preds, 'y':y}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        print(f'Validation\nAccuracy {avg_acc}\n')
        self.log('val_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_acc', avg_acc, on_step=False, on_epoch=True, prog_bar=False)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
        

        # print(outputs.shape, y_mask.shape)
if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0)
    val_fold_idx = parser.parse_args().fold
    
    # Load the configuration
    config = load_config_as_dict("utils/config_seg.yaml").get_dict()
    print(config)
    torch.manual_seed(config['seed'])
    train_dataset = SleepSegmentationDataset(data_path=config['data_dir'], split='train')
    val_dataset = SleepSegmentationDataset(data_path=config['data_dir'], split='val')
    print(f'NO. training samples: {len(train_dataset)}')
    print(f'NO. validation samples: {len(val_dataset)}')
    group_name = f"unet_v0"
    wandb_logger = WandbLogger(project="NeuroTech", group=group_name)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=8, persistent_workers=True)

    # Instantiate the trainer
    model = NeuroTech_Trainer(config)    
    checkpoint_callback = ModelCheckpoint(
        monitor = 'val_acc',
        mode = 'max',
        dirpath = f'ckpt/{group_name}',
        filename = f'{config["dataset"]}-{config["model_name"]}-' + '{epoch:02d}-{val_acc:.2f}' # rand/curriculum
    )
    trainer = pl.Trainer(max_epochs=config["epochs"], logger=wandb_logger,
                         accelerator='gpu', 
                         gpus=1,
                         num_sanity_val_steps=0,
                         precision=16,
                         log_every_n_steps=1,
                         callbacks=[checkpoint_callback])
    
    trainer.fit(model, train_loader, val_loader)
    