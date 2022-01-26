import torch.nn as nn
import torch.nn.functional as F

def _get_activation(activation='relu'):
    if activation == 'relu':
        return nn.ReLU()
    elif activation is None:
        return nn.Identity()

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