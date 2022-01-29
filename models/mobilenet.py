import torch
import torch.nn as nn

from models.common import _conv2d_bn_act


class ReLU6(nn.Module):
    
    def __init__(self):
        super(ReLU6, self).__init__()
        pass
    
    def forward(self, x):
        return torch.minimum( torch.maximum(0, x), 6 )


class MBConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, downsampling=False, expansion=1):
        super(MBConv, self).__init__()
        
        self._expansion = expansion
        self._stride = 2 if downsampling else 1
        self._downsampling = downsampling
        
        self.block = nn.Sequential(
            _conv2d_bn_act(activation='relu6', in_channels=in_channels, out_channels=in_channels * self._expansion, kernel_size=1),
            _conv2d_bn_act(activation='relu6', in_channels=in_channels * self._expansion, out_channels=in_channels * self._expansion,\
                           kernel_size=3, padding=1, stride=self._stride, groups=in_channels * self._expansion),
            _conv2d_bn_act(activation=None, in_channels=in_channels * self._expansion, out_channels=out_channels, kernel_size=1, bias=False)
        )
        
        self._skip_connection = (not downsampling and in_channels == out_channels)
    
    def forward(self, x):
        if self._skip_connection:
            identity = x
        
        x = self.block(x)
        
        if self._skip_connection:
            x = x + identity
        
        return x


class MobileNetV2(nn.Module):
    
    def __init__(self, n_classes, n_channels):
        super(MobileNetV2, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            _conv2d_bn_act(activation='relu6', in_channels=n_channels, out_channels=32, kernel_size=3, padding=1, stride=2),
            MBConv(32, 16),
            MBConv(16, 24, True, 6),
            MBConv(24, 24, False, 6),
            MBConv(24, 32, True, 6),
            *[MBConv(32, 32, False, 6) for i in range(2)],
            MBConv(32, 64, True, 6),
            *[MBConv(64, 64, False, 6) for i in range(3)],
            MBConv(64, 96, False, 6),
            *[MBConv(96, 96, False, 6) for i in range(2)],
            MBConv(96, 160, True, 6),
            *[MBConv(160, 160, False, 6) for i in range(2)],
            MBConv(160, 320, False, 6),
            _conv2d_bn_act(activation='relu6', in_channels=320, out_channels=1280, kernel_size=1)
        )
        
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        
        self.fc_out   = nn.Linear(1280, n_classes)
        
    def forward(self, x):
        # feature extraction
        x = self.feature_extractor(x)
        
        # classification
        print(x.size())
        x = self.avg_pool(x).squeeze()
        x = self.fc_out(x)
        
        return x