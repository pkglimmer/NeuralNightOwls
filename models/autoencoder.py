import torch
import torch.nn as nn
import pytorch_lightning as pl

class res_1d(nn.Module):
    def __init__(self, io_ch, ks=3):
        super(res_1d, self).__init__()
        assert (ks - 1) % 2 == 0
        pd = int((ks - 1) / 2)
        self.res1_3 = nn.Sequential(
            nn.Conv1d(io_ch, io_ch, ks, padding=pd),
            nn.ReLU(True),
            nn.Conv1d(io_ch, io_ch, ks, padding=pd))
        self.res_relu5 = nn.ReLU(True)

    def forward(self, x):
        return self.res_relu5(x + self.res1_3(x))

class Encoder(pl.LightningModule):
    def __init__(self, in_channels, out_channels, kernel_size, num_layers, stride=1, padding=0):
        super(Encoder, self).__init__()
        layers = []
        self.out_dim = 125 # downsampling 1/8
        mode = 'AE'
        channels = [1, 16, 32, 4]
        relu_slope = 0.2
        for i in range(num_layers):
            if mode == 'AE':
                layers.append(torch.nn.Conv1d(channels[i], channels[i+1], kernel_size, stride, padding))
                layers.append(nn.BatchNorm1d(channels[i+1]))
                layers.append(nn.LeakyReLU(negative_slope=relu_slope))
                layers.append(res_1d(channels[i+1]))
                layers.append(torch.nn.MaxPool1d(2, ))
            else:
                layers.append(torch.nn.Conv1d(in_channels if i == 0 else out_channels, 
                                            out_channels, kernel_size, stride, padding))
                layers.append(nn.BatchNorm1d(out_channels))
                layers.append(nn.ReLU())
                layers.append(res_1d(out_channels))
                layers.append(torch.nn.MaxPool1d(2, ))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x
    
def print_output_dim(module, input, output):
    print(f"{module} input: {input[0].shape} \n output shape: {output.shape}")
    
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_layers, stride=1, padding=0):
        super(Decoder, self).__init__()
        layers = []
        mode = 'AE'
        channels = [4, 16, 32, 1]
        relu_slope = 1e-2
        for i in range(num_layers):
            if mode == 'AE':
                layers.append(nn.ConvTranspose1d(channels[i], channels[i+1], kernel_size, stride, padding, output_padding=1))
                layers.append(nn.BatchNorm1d(channels[i+1]))
                layers.append(nn.LeakyReLU(negative_slope=relu_slope))
                layers.append(res_1d(channels[i+1]))
                layers.append(nn.LeakyReLU(negative_slope=relu_slope))
            else:
                layers.append(nn.ConvTranspose1d(in_channels, 
                                                out_channels if i==num_layers-1 else in_channels, 
                                                kernel_size, stride, padding, output_padding=1))
                layers.append(nn.BatchNorm1d(out_channels if i==num_layers-1 else in_channels))
                layers.append(nn.ReLU())
        # layers.append(res_1d(out_channels))
        layers.append(torch.nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.layers = nn.Sequential(*layers)
        # for name, layer in self.layers.named_children():
        #     layer.register_forward_hook(print_output_dim)

    def forward(self, x):
        x = self.layers(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self, encoder_params, decoder_params, return_z = False):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(**encoder_params)
        self.decoder = Decoder(**decoder_params)
        self.return_z = return_z
        
    def forward(self, x):
        z_e = self.encoder(x)
        x_hat = self.decoder(z_e)
        if self.return_z:
            return z_e, x_hat
        return x_hat
    
if __name__ == "__main__":
    encoder_params = {'in_channels': 1, 'out_channels': 32, 'kernel_size': 3, 'num_layers': 3, 'stride': 1, 'padding': 1}
    decoder_params = {'in_channels': 32, 'out_channels': 1, 'kernel_size': 3, 'num_layers': 3, 'stride': 2, 'padding': 1}
    model = AutoEncoder(encoder_params, decoder_params, return_z=True)

    # Dummy input
    batch_size = 4
    seq_length = 3840
    x = torch.randn(batch_size, 1, seq_length)

    # Forward pass
    z, x_hat = model(x)

    # print("Reconstruction: ", x_hat)
    # print("Loss: ", loss)
    print(f'x.shape {x.shape}')
    print(f'x_hat.shape {x_hat.shape}')