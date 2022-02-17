import torch
import torch.nn as nn

from models.common import _get_activation


def _conv3x3_act(in_channels, out_channels, padding=1, activation='relu'):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=padding),
        _get_activation(activation)
    )


def _deconv2x2_act(in_channels, out_channels, activation='relu'):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
        _get_activation(activation)
    )


class UNetEncoderBlock(nn.Module):
    
    def __init__(self, in_channels, n_filters, dropout=0):
        super().__init__()
        
        self.conv = nn.Sequential(
            _conv3x3_act(in_channels, n_filters),
            _conv3x3_act(n_filters, n_filters),
        )
        
        self.max_pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.conv(x)
        x_out = self.dropout(self.max_pool(x))
        
        return x_out, x


class UNetDecoderBlock(nn.Module):
    
    def __init__(self, in_channels, n_filters, dropout=0):
        super().__init__()
        
        self.deconv = _deconv2x2_act(in_channels, n_filters)
        self.dropout = nn.Dropout(dropout)
        
        self.conv = nn.Sequential(
            _conv3x3_act(n_filters * 2, n_filters),
            _conv3x3_act(n_filters, n_filters),
        )
    
    def forward(self, x, x_skip):
        x = self.dropout(self.deconv(x))
        x = torch.cat((x_skip, x), dim=1)
        x = self.conv(x)
        
        return x


class UNet(nn.Module):
    
    def __init__(self, n_channels, n_classes):
        super().__init__()
        
        self.encoder_1 = UNetEncoderBlock(n_channels, 64)
        self.encoder_2 = UNetEncoderBlock(64, 128)
        self.encoder_3 = UNetEncoderBlock(128, 256)
        self.encoder_4 = UNetEncoderBlock(256, 512)
        
        self.bottleneck = nn.Sequential(
            _conv3x3_act(512, 1024),
            _conv3x3_act(1024, 1024)
        )
        
        self.decoder_1 = UNetDecoderBlock(1024, 512)
        self.decoder_2 = UNetDecoderBlock(512, 256)
        self.decoder_3 = UNetDecoderBlock(256, 128)
        self.decoder_4 = UNetDecoderBlock(128, 64)
        
        self.conv_out  = nn.Conv2d(64, n_classes, kernel_size=1)
        
    def forward(self, x):
        x, x_e1 = self.encoder_1(x)
        x, x_e2 = self.encoder_2(x)
        x, x_e3 = self.encoder_3(x)
        x, x_e4 = self.encoder_4(x)
        
        x = self.bottleneck(x)
        
        x = self.decoder_1(x, x_e4)
        x = self.decoder_2(x, x_e3)
        x = self.decoder_3(x, x_e2)
        x = self.decoder_4(x, x_e1)
        
        x = self.conv_out(x)
        
        return x