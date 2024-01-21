from models.resnet1d import *
# from resnet1d import *

class ConvDecoder(nn.Module):
    def __init__(self, out_channels = 5, out_len = 3840):
        super(ConvDecoder, self).__init__()
        # channels = [16, 32, 64, 16]
        self.out_len = out_len # 3840 for MSG, 2400 for BCM
        channels = [32, 64, 128, 64]
        self.out_channels = out_channels
        self.conv1 = nn.ConvTranspose1d(64, channels[0], kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(channels[0])
        self.conv2 = nn.ConvTranspose1d(channels[0], channels[1], kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(channels[1])
        self.conv3 = nn.ConvTranspose1d(channels[1], channels[2], kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(channels[2])
        self.conv4 = nn.ConvTranspose1d(channels[2], channels[3], kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(channels[3])
        self.conv5 = nn.ConvTranspose1d(channels[3], self.out_channels, kernel_size=4, stride=2, padding=1)
        # nn.init.kaiming_normal_(self.conv1.weight)
        # nn.init.constant_(self.conv1.bias, 0.0)
        # nn.init.kaiming_normal_(self.conv2.weight)
        # nn.init.constant_(self.conv2.bias, 0.0)
        # nn.init.kaiming_normal_(self.conv3.weight)
        # nn.init.constant_(self.conv3.bias, 0.0)
        # nn.init.kaiming_normal_(self.conv4.weight)
        # nn.init.constant_(self.conv4.bias, 0.0)
        # nn.init.kaiming_normal_(self.conv5.weight)
        # nn.init.constant_(self.conv5.bias, 0.0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        return x.view(-1, self.out_channels, self.out_len)


if __name__ == "__main__":
    encoder = resnet18(num_input_channels=5, num_classes=2, return_feature=True)
    decoder = ConvDecoder()
    dummy = torch.zeros(1, 1, 19200)
    x, z = encoder(dummy)
    print(f'encoder_out.shape {x.shape}')
    x = decoder(x)
    print(f'decoder_out.shape {x.shape}')