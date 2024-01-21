import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class UNet1D(nn.Module):
    def __init__(self):
        super(UNet1D, self).__init__()
        
        # Encoder
        self.down_conv1 = ConvBlock(6, 32)
        self.down_conv2 = ConvBlock(32, 64)
        self.down_conv3 = ConvBlock(64, 128)

        # Decoder
        self.up_conv3 = ConvBlock(128 + 64, 64)
        self.up_conv2 = ConvBlock(64 + 32, 32)
        self.final_conv = nn.Conv1d(32, 6, kernel_size=1)
        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        x1 = self.down_conv1(x)
        x2 = self.pool(x1)
        x3 = self.down_conv2(x2)
        x4 = self.pool(x3)
        x5 = self.down_conv3(x4)

        # Decoder
        x = F.interpolate(x5, scale_factor=2, mode='linear', align_corners=True)
        x = torch.cat([x, x3], 1)  # Skip connection
        x = self.up_conv3(x)
        
        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        x = torch.cat([x, x1], 1)  # Skip connection
        x = self.up_conv2(x)

        x = self.final_conv(x)
        x = torch.squeeze(x, 1) # To match the desired output shape (N,)
        return x

if __name__ == '__main__':
    model = UNet1D().cuda()
    input = torch.randn(32, 6, 30 * 3000).cuda() # Batch size of 32, 6 channels, 3000 samples
    print(input.shape)
    output = model(input)
    print(output.shape)  # Expected output shape: (32, N)
