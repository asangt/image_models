import torch.nn as nn

from .common import _get_activation, ZeroPadShortcut

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

class ResBlock(nn.Module):
    _expansion = 1
    
    def __init__(self, in_channels, out_channels, downsampling=False, shortcut='projection', activation='relu', composition='original'):
        super(ResBlock, self).__init__()
        
        self._activation = activation
        self._stride = 2 if downsampling else 1
        
        self.act = None
        
        if composition == 'original':
            self.block = nn.Sequential(
                _conv2d_bn_act(activation=activation, in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=self._stride, padding=1),
                _conv2d_bn_act(activation=None, in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
            )
            
            self.act = _get_activation(activation)
        elif composition == 'bn post-addition':
            self.block = nn.Sequential(
                _conv2d_bn_act(activation=activation, in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=self._stride, padding=1),
                _conv2d_bn_act(activation=None, batch_norm=False, in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
            )
            
            self.act = nn.Sequential(
                nn.BatchNorm2d(out_channels),
                _get_activation(activation)
            )
        elif composition == 'activation pre-addition':
            self.block = nn.Sequential(
                _conv2d_bn_act(activation=activation, in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=self._stride, padding=1),
                _conv2d_bn_act(activation=activation, in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
            )
        elif composition == 'full pre-activation':
            self.block = nn.Sequential(
                _conv2d_bn_act('pre-act', activation, in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=self._stride, padding=1),
                _conv2d_bn_act('pre-act', activation, in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
            )
        elif composition == 'partial pre-activation':
            self.block = nn.Sequential(
                _get_activation(activation),
                _conv2d_bn_act(activation=activation, in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=self._stride, padding=1),
                _conv2d_bn_act(activation=None, in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
            )
        else:
            raise ValueError('Block composition should be either `original`, `bn post-addition`, `activation pre-addition`, `full pre-activation` or `partial pre-activation`.')
            
        if not downsampling and shortcut != 'full projection':
            self.shortcut = nn.Identity()
        elif shortcut == 'projection' or shortcut == 'full projection':
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=self._stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        elif shortcut == 'zero-pad':
            self.shortcut = ZeroPadShortcut(in_channels, stride=self._stride)
        else:
            raise ValueError('Block shortcut should be either `projection`, `full projection` or `zero-pad`.')
        
        self._init_weights()
    
    def _init_weights(self):
        _count_bns = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=self._activation)
            elif isinstance(m, nn.BatchNorm2d):
                _count_bns += 1
                if _count_bns == 2:
                    nn.init.constant_(m.weight, 0)
                else:
                    nn.init.constant_(m.weight, 1)
                
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        out = self.block(x)
        out = self.act(out + self.shortcut(x))
        
        return out

class BottleneckBlock(nn.Module):
    _expansion = 4
    
    def __init__(self, in_channels, out_channels, downsampling=False, shortcut='projection', activation='relu', composition='original'):
        super(BottleneckBlock, self).__init__()
        
        self._activation = activation
        self._stride = 2 if downsampling else 1
        
        self.act = None
        
        if composition == 'original':
            self.block = nn.Sequential(
                _conv2d_bn_act(activation=activation, in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=self._stride, padding=0),
                _conv2d_bn_act(activation=activation, in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
                _conv2d_bn_act(activation=None, in_channels=out_channels, out_channels=out_channels * self._expansion, kernel_size=1, padding=0)
            )
            
            self.act = _get_activation(activation)
        elif composition == 'bn post-addition':
            self.block = nn.Sequential(
                _conv2d_bn_act(activation=activation, in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=self._stride, padding=0),
                _conv2d_bn_act(activation=activation, in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
                _conv2d_bn_act(activation=None, batch_norm=False, in_channels=out_channels, out_channels=out_channels * self._expansion, kernel_size=1, padding=0)
            )
            
            self.act = nn.Sequential(
                nn.BatchNorm2d(out_channels * self._expansion),
                _get_activation(activation)
            )
        elif composition == 'activation pre-addition':
            self.block = nn.Sequential(
                _conv2d_bn_act(activation=activation, in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=self._stride, padding=0),
                _conv2d_bn_act(activation=activation, in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
                _conv2d_bn_act(activation=activation, in_channels=out_channels, out_channels=out_channels * self._expansion, kernel_size=1, padding=0)
            )
        elif composition == 'full pre-activation':
            self.block = nn.Sequential(
                _conv2d_bn_act('pre-act', activation, in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=self._stride, padding=0),
                _conv2d_bn_act('pre-act', activation, in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
                _conv2d_bn_act('pre-act', activation, in_channels=out_channels, out_channels=out_channels * self._expansion, kernel_size=1, padding=0)
            )
        elif composition == 'partial pre-activation':
            self.block = nn.Sequential(
                _get_activation(activation),
                _conv2d_bn_act(activation=activation, in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=self._stride, padding=0),
                _conv2d_bn_act(activation=activation, in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
                _conv2d_bn_act(activation=None, in_channels=out_channels, out_channels=out_channels * self._expansion, kernel_size=1, padding=0)
            )
        else:
            raise ValueError('Block composition should be either `original`, `bn post-addition`, `activation pre-addition`, `full pre-activation` or `partial pre-activation`.')
        
        if shortcut == 'projection' or shortcut == 'full projection':
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self._expansion, kernel_size=1, stride=self._stride, bias=False),
                nn.BatchNorm2d(out_channels * self._expansion)
            )
        elif shortcut == 'zero-pad':
            self.shortcut = ZeroPadShortcut(out_channels * (self._expansion - 1), stride=self._stride)
        else:
            raise ValueError('Block shortcut should be either `projection`, `full projection` or `zero-pad`.')
        
        self._init_weights()
    
    def _init_weights(self):
        _count_bns = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=self._activation)
            elif isinstance(m, nn.BatchNorm2d):
                _count_bns += 1
                if _count_bns == 3:
                    nn.init.constant_(m.weight, 0)
                else:
                    nn.init.constant_(m.weight, 1)
                
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        out = self.block(x)
        out = self.act(out + self.shortcut(x))
        
        return out

class ResNet(nn.Module):
    def __init__(self, n_classes, n_channels, model_structure, block_type='resblock', block_composition='original', block_shortcut='projection'):
        super(ResNet, self).__init__()
        
        self._activation = 'relu'
        
        if block_type not in ['resblock', 'bottleneck']:
            raise ValueError('Block type should be either `resblock` or `bottleneck`.')
        
        if block_composition not in ['original', 'bn post-addition', 'activation pre-addition', 'full pre-activation', 'partial pre-activation']:
            raise ValueError('Block composition should be either `original`, `bn post-addition`, `activation pre-addition`, `full pre-activation` or `partial pre-activation`.')
        
        if block_shortcut not in ['projection', 'full projection', 'zero-pad']:
            raise ValueError('Block shortcut should be either `projection`, `full projection` or `zero-pad`.')
        
        _block = ResBlock if block_type == 'resblock' else BottleneckBlock
        
        self.conv1 = _conv2d_bn_act(activation=self._activation, in_channels=n_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        
        self.conv2_x = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            _block(64, 64, False, block_shortcut, self._activation, block_composition),
            *[_block(64 * _block._expansion, 64, False, block_shortcut, self._activation, block_composition) for i in range(model_structure[0] - 1)]
        )
        
        self.conv3_x = nn.Sequential(
            _block(64 * _block._expansion, 128, True, block_shortcut, self._activation, block_composition),
            *[_block(128 * _block._expansion, 128, False, block_shortcut, self._activation, block_composition) for i in range(model_structure[1] - 1)]
        )
        
        self.conv4_x = nn.Sequential(
            _block(128 * _block._expansion, 256, True, block_shortcut, self._activation, block_composition),
            *[_block(256 * _block._expansion, 256, False, block_shortcut, self._activation, block_composition) for i in range(model_structure[2] - 1)]
        )
        
        self.conv5_x = nn.Sequential(
            _block(256 * _block._expansion, 512, True, block_shortcut, self._activation, block_composition),
            *[_block(512 * _block._expansion, 512, False, block_shortcut, self._activation, block_composition) for i in range(model_structure[3] - 1)]
        )

        self.feature_extractor = nn.Sequential(
            self.conv1,
            self.conv2_x,
            self.conv3_x,
            self.conv4_x,
            self.conv5_x
        )
        
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        
        self.fc_out = nn.Linear(512 * _block._expansion, n_classes)
        
    
    def forward(self, x):
        # feature extraction
        x = self.feature_extractor(x)
        
        # classificator
        x = self.avg_pool(x).squeeze()
        x = self.fc_out(x)
        
        return x

# Convenient wrappers for the most common ResNet architectures introduced in the original paper

def _build_resnet(model_name, n_classes, n_channels, block_type, block_composition, block_shortcut):
    model_names = {
        'resnet18' : [2, 2, 2, 2],
        'resnet34' : [3, 4, 6, 3],
        'resnet50' : [3, 4, 6, 3],
        'resnet101' : [3, 4, 23, 3],
        'resnet152' : [3, 8, 36, 3]
    }
    model_structure = model_names[model_name]

    return ResNet(n_classes, n_channels, model_structure, block_type, block_composition, block_shortcut)

def resnet18(n_classes, n_channels, block_composition='original', block_shortcut='projection'):
    return _build_resnet('resnet18', n_classes, n_channels, 'resblock', block_composition, block_shortcut)

def resnet34(n_classes, n_channels, block_composition='original', block_shortcut='projection'):
    return _build_resnet('resnet34', n_classes, n_channels, 'resblock', block_composition, block_shortcut)

def resnet50(n_classes, n_channels, block_composition='original', block_shortcut='projection'):
    return _build_resnet('resnet50', n_classes, n_channels, 'bottleneck', block_composition, block_shortcut)

def resnet101(n_classes, n_channels, block_composition='original', block_shortcut='projection'):
    return _build_resnet('resnet101', n_classes, n_channels, 'bottleneck', block_composition, block_shortcut)

def resnet152(n_classes, n_channels, block_composition='original', block_shortcut='projection'):
    return _build_resnet('resnet152', n_classes, n_channels, 'bottleneck', block_composition, block_shortcut)