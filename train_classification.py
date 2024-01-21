import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint#, TQDMProgressBar
import torch.nn.functional as F
from torchsampler import ImbalancedDatasetSampler
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from SleepDataset import SleepDataset
from models.resnet1d import resnet18, resnet10, resnext50_32x4d, resnet34, resnet50
from models.cnn_lstm import LSTMClassifier
from models.dagmm import DaGMM_wrapper
from utils.mixup_utils import mixup_data, mixup_criterion
import argparse
import os
import warnings
from utils.load_config import load_config_as_dict
warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
model_dict = {
    'R10': resnet10(num_input_channels=6, num_classes=6),
    'R18': resnet18(num_input_channels=6, num_classes=6), # main
    'R34': resnet34(num_input_channels=6, num_classes=6),
    'R50': resnet50(num_input_channels=6, num_classes=6),
    'X50': resnext50_32x4d(num_input_channels=6, num_classes=6),
    'res50': resnext50_32x4d(num_input_channels=6, num_classes=6),
    'lstm': LSTMClassifier(in_dim=6, hidden_size=32, num_layers=4, dropout=0.1, n_classes=6, bidirectional=True),
}

class NeuroTech_Trainer(pl.LightningModule):
    def __init__(self, args):
        super(NeuroTech_Trainer, self).__init__()
        self.hparams.update(args)
        self.model = model_dict[args['model_name']]
        self.save_hyperparameters()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x.half())

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.type(torch.LongTensor).cuda()
        if self.hparams['mixup']:
            x, y_a, y_b, lambda_ = mixup_data(x, y, alpha=0.75)
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1) # size (B, num_classes=2)
        _, preds = torch.max(probs, dim=1)
        if self.hparams['mixup']:
            loss = mixup_criterion(self.criterion, logits, y_a, y_b, lambda_)
            self.log('train_loss', loss)
            return {'loss': loss, 'probs':probs, 'y':y}
        else:
            loss = self.criterion(logits, y)
            acc = (preds == y).float().mean()
            self.log('train_acc', acc)
            self.log('train_loss', loss)
            return {'loss': loss, 'train_acc': acc, 'probs':probs, 'y':y}

    def training_epoch_end(self, outputs):
        avg_loss = torch.tensor([x['loss'] for x in outputs]).mean()
        self.log('train_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=False)
        if not self.hparams['mixup']:
            probs = torch.cat([x['probs'] for x in outputs], dim = 0)
            avg_acc = torch.stack([x['train_acc'] for x in outputs]).mean()
            y = torch.cat([x['y'] for x in outputs], dim=0)
            _, preds = torch.max(probs, dim=1)
            auc = roc_auc_score(y.detach().cpu().numpy(), probs.detach().cpu().numpy(), multi_class='ovo')
            precision, recall, f1_score, support = precision_recall_fscore_support(y.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=1)
            # print(f'\ntotal train seizure: {sum(y).item()}, TP: {sum((preds==1) & (y==1))}, FN: {sum((preds==0) & (y==1))}, \
            #     TN: {sum((preds==0) & (y==0))}, FP: {sum((preds==1) & (y==0))}\n')
            print(f'accuracy {avg_acc}, auc {auc}, precision {precision}, recall {recall}, f1 {f1_score}\n')
            self.log('train_acc', avg_acc, on_step=False, on_epoch=True, prog_bar=False)    
            self.log('train_AUROC', auc, on_step=False, on_epoch=True, prog_bar=False)
            self.log('train_prec', precision, on_step=False, on_epoch=True, prog_bar=False)
            self.log('train_recall', recall, on_step=False, on_epoch=True, prog_bar=False)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.type(torch.LongTensor).cuda()
        logits = self.forward(x)
        # Convert the logits to probabilities using softmax
        probs = F.softmax(logits, dim=1)
        _, preds = torch.max(probs, dim=1)
        acc = (preds == y).float().mean()
        loss = nn.CrossEntropyLoss()(logits, y)
        return {'val_loss': loss, 'val_acc': acc, 'probs':probs, 'y':y}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        probs = torch.cat([x['probs'] for x in outputs], dim = 0)
        y = torch.cat([x['y'] for x in outputs], dim = 0)
        _, preds = torch.max(probs, dim=1)
        print(f'\nVal seizure: {sum(y).item()}, TP: {sum((preds==1) & (y==1))}, FN: {sum((preds==0) & (y==1))}, \
              TN: {sum((preds==0) & (y==0))}, FP: {sum((preds==1) & (y==0))}\n\n')
        auc = roc_auc_score(y.cpu().numpy(), probs.cpu().numpy(), multi_class='ovo')
        precision, recall, f1_score, support = precision_recall_fscore_support(y.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=1)
        print(f'Validation\nAccuracy {avg_acc},\n AUC: {auc},\n PRECISION {precision}, RECALL {recall}, F1 {f1_score}')
        self.log('val_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_acc', avg_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_AUROC', auc, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_prec', precision, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_recall', recall, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_f1', f1_score, on_step=False, on_epoch=True, prog_bar=False)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
        

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=0)
    val_fold_idx = parser.parse_args().fold
    
    # Load the configuration
    config = load_config_as_dict("utils/config_cls.yaml").get_dict()
    print(config)
    torch.manual_seed(config['seed'])
    
    train_dataset, val_dataset = SleepDataset(os.path.join(config['data_dir'], 'train_data.npy'), os.path.join(config['data_dir'], 'train_labels.npy')), \
                                    SleepDataset(os.path.join(config['data_dir'], 'val_data.npy'), os.path.join(config['data_dir'], 'val_labels.npy'))  
    print(f'NO. training samples: {len(train_dataset)}')
    print(f'NO. validation samples: {len(val_dataset)}')
    group_name = f"sleep_v1_balanced"
    wandb_logger = WandbLogger(project="NeuroTech", group=group_name)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=8, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=8, persistent_workers=True)
    if config['balanced_sampler'] and not config['undersample']:
        sampler = ImbalancedDatasetSampler(train_dataset)
        train_loader = DataLoader(train_dataset, sampler=sampler, batch_size=config['batch_size'], num_workers=8, persistent_workers=True)


    # Instantiate the trainer
    model = NeuroTech_Trainer(config)
    if config['pretrained']:
        ckpt_path = config['weights_dir']
        recon_state_dict = torch.load(ckpt_path, map_location='cuda')
        new_state_dict = {}
        for key, value in recon_state_dict['encoder'].items():
            if key.startswith('encoder'):
                new_state_dict[key.replace('encoder', 'model')] = value
        model.load_state_dict(new_state_dict)
    
    checkpoint_callback = ModelCheckpoint(
        monitor = 'val_acc',
        mode = 'max',
        dirpath = f'ckpt/{config["model_name"]}',
        filename = f'{config["dataset"]}-' + '{epoch:02d}-{val_acc:.2f}' # rand/curriculum
    )
    trainer = pl.Trainer(max_epochs=config["epochs"], logger=wandb_logger,
                         accelerator='gpu', 
                         gpus=1,
                         num_sanity_val_steps=0,
                         precision=16,
                         log_every_n_steps=1,
                         callbacks=[checkpoint_callback])
    
    trainer.fit(model, train_loader, val_loader)
    