import torch
import torch.nn as nn
import torch.nn.functional as F


class ReLU6(nn.Module):
    
    def __init__(self):
        super(ReLU6, self).__init__()
        pass
    
    def forward(self, x):
        return torch.minimum( torch.maximum(0, x), 6 )


def _get_activation(activation='relu'):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'relu6':
        return ReLU6()
    elif activation is None:
        return nn.Identity()

    
def _conv2d_bn_act(mode='standard', activation='relu', batch_norm=True, **kwargs):
    if mode == 'standard':
        return nn.Sequential(
            nn.Conv2d(**kwargs),
            nn.BatchNorm2d(kwargs['out_channels']) if batch_norm else nn.Identity(),
            _get_activation(activation)
        )
    elif mode == 'pre-act':
        return nn.Sequential(
            nn.BatchNorm2d(kwargs['in_channels']) if batch_norm else nn.Identity(),
            _get_activation(activation),
            nn.Conv2d(**kwargs)
        )


class ZeroPadShortcut(nn.Module):
    def __init__(self, n_channels, stride=1):
        super(ZeroPadShortcut, self).__init__()

        self._n_channels = n_channels
        self._stride = stride
    
    def forward(self, x):
        x = F.pad(x, (0, 0, 0, 0, self._n_channels, 0))
        batch_size, channels, height, width = x.size()
        x = x.resize_(batch_size, channels, height // self._stride, width // self._stride)
        
        return x