import torch
from torch import nn
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.LeakyReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out
    
def create_encoder(in_channels, out_dim=32):
    dims = [8, 16, 32, 32, 16, 8, 64]
    encoder = nn.Sequential(
        ResidualBlock(in_channels, dims[0]),
        nn.MaxPool1d(5),
        ResidualBlock(dims[0], dims[1]),
        nn.MaxPool1d(3),
        ResidualBlock(dims[1], dims[2]),
        nn.MaxPool1d(2),
        ResidualBlock(dims[2], dims[3]),
        nn.MaxPool1d(2),
        ResidualBlock(dims[3], dims[4]),
        nn.MaxPool1d(2),
        ResidualBlock(dims[4], dims[5]),
        nn.AdaptiveAvgPool1d(dims[6]),
    )
    fc = nn.Sequential(
        nn.Linear(dims[5] * dims[6], out_dim),
        nn.LeakyReLU(inplace=True)
    )
    return encoder, fc

class CNN_late_fusion(pl.LightningModule):
    def __init__(self, n_class):
        super(CNN_late_fusion, self).__init__()
        out_dims = {'eda':64, 'hr':16, 'acc':64, 'bvp':64, 'temp':16}
        self.encoder_eda, self.fc_eda = create_encoder(1, out_dims['eda'])
        self.encoder_hr, self.fc_hr = create_encoder(1, out_dims['hr'])
        self.encoder_acc, self.fc_acc = create_encoder(1, out_dims['acc'])
        self.encoder_bvp, self.fc_bvp = create_encoder(1, out_dims['bvp'])
        self.encoder_temp, self.fc_temp = create_encoder(1, out_dims['temp'])
        fused_dim = sum(out_dims.values())
        self.cls = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, n_class)
        )

    def forward(self, x):
        acc, bvp, eda, hr, temp = x
        B = acc.shape[0]
        # feat_eda = self.fc_eda(self.encoder_eda(eda).squeeze(-1))
        # feat_hr = self.fc_hr(self.encoder_hr(hr).squeeze(-1))
        # feat_bvp = self.fc_bvp(self.encoder_bvp(bvp).squeeze(-1))
        # feat_acc = self.fc_acc(self.encoder_acc(acc).squeeze(-1))
        # feat_temp = self.fc_temp(self.encoder_temp(temp).squeeze(-1))
        feat_eda = self.fc_eda(self.encoder_eda(eda).reshape(B, -1))
        feat_hr = self.fc_hr(self.encoder_hr(hr).reshape(B, -1))
        feat_bvp = self.fc_bvp(self.encoder_bvp(bvp).reshape(B, -1))
        feat_acc = self.fc_acc(self.encoder_acc(acc).reshape(B, -1))
        feat_temp = self.fc_temp(self.encoder_temp(temp).reshape(B, -1))
        feats = torch.squeeze(torch.cat([feat_eda, feat_hr, feat_bvp, feat_acc, feat_temp], dim=1), dim=-1)
        out = self.cls(feats)
        return out


def main():
    # verify the models with dummy tensor
    model = create_encoder(8)
    # model = model_ResNet(layers=[2,2,2,2], num_classes=5)
    dummy = torch.randn((10, 8, 19200)) # PTB-XL
    print(f'total params: {sum(p.numel() for p in model.parameters())}')
    print(f'dummy.shape {dummy.shape}')
    out = model(dummy)
    print(f'out.shape {out.shape}')
    
if __name__ == "__main__":
    msg_sample = torch.zeros(1, 9, 76800)
    acc = msg_sample[:, 4, ::4].unsqueeze(0)
    bvp = msg_sample[:, 5, ::2].unsqueeze(0)
    eda = msg_sample[:, 6, ::32].unsqueeze(0)
    hr = msg_sample[:, 7, ::32].unsqueeze(0)
    temp = msg_sample[:, 8, ::128].unsqueeze(0)
    x = (acc, bvp, eda, hr, temp)
    print(acc.shape)
    model = CNN_late_fusion(n_class=2)
    y = model(x)
    print(f"output.shape {y.shape}")

    