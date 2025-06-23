import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import os
import math

from config import opt


model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

cfgs = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VggNet(nn.Module):
    def __init__(self, cfg_key=None, features=None, num_classes=1000, batch_norm=False, in_channels=3, init_weights=True):
        """
        VggNet类初始化
        参数:
            cfg_key: 使用配置键('D'或'E')构建网络
            features: 直接提供预构建的特征提取层
            num_classes: 分类数量
            batch_norm: 是否使用批归一化
            in_channels: 输入通道数
            init_weights: 是否初始化权重
        """
        super(VggNet, self).__init__()
        
        # 如果提供了features，直接使用；否则根据cfg_key构建
        if features is not None:
            self.features = features
        else:
            assert cfg_key is not None, "必须提供cfg_key或features"
            self.features = self._make_layers(cfgs[cfg_key], batch_norm, in_channels)
            
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def _make_layers(self, cfg, batch_norm=False, in_channels=3):
        layers = []
        input_channels = in_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(input_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                input_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)



def vggnet(model_name, pretrained=False, pretrained_weight=None, **kwargs):

    if model_name == 'vgg16':
        cfg_key = 'D'
        arch = 'vgg16'
    elif model_name == 'vgg19':
        cfg_key = 'E'
        arch = 'vgg19'
    else:
        raise Exception('Unsupported VGG model: ', model_name)
    
    kwargs['init_weights'] = not pretrained
    in_channels = 3 if opt.without_mask else 4
    batch_norm = False
    
    model = VggNet(cfg_key=cfg_key, batch_norm=batch_norm, in_channels=in_channels, **kwargs)
    
    if pretrained:
        state_dict = None
        if pretrained_weight:
            state_dict = torch.load(pretrained_weight)
        elif arch in model_urls:
            if not os.path.exists(opt.pretrained_model_path):
                os.makedirs(opt.pretrained_model_path)
            state_dict = model_zoo.load_url(model_urls[arch], model_dir=opt.pretrained_model_path)
        else:
            assert False, f"model_urls not given {arch}."

        if state_dict:
            model_dict = model.state_dict()

            if in_channels == 4:
                conv1_3ch = state_dict['features.0.weight']
                conv1_4th = (0.299 * conv1_3ch[:, 0, :, :] +
                             0.587 * conv1_3ch[:, 1, :, :] +
                             0.114 * conv1_3ch[:, 2, :, :]).unsqueeze(1)
                state_dict['features.0.weight'] = torch.cat((conv1_3ch, conv1_4th), dim=1)

            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
            
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(f'loaded pretrained {arch} from {pretrained_weight if pretrained_weight else model_urls[arch]}')
    
    return model

def vggnet_for_depth(model_name, pretrained=False, pretrained_weight=None, **kwargs):
    if model_name == 'vgg16':
        cfg_key = 'D'
        arch = 'vgg16'
    elif model_name == 'vgg19':
        cfg_key = 'E'
        arch = 'vgg19'
    else:
        raise Exception('Unsupported VGG model: ', model_name)
    
    kwargs['init_weights'] = not pretrained
    in_channels = 2
    batch_norm = False
    
    model = VggNet(cfg_key=cfg_key, batch_norm=batch_norm, in_channels=in_channels, **kwargs)
    
    if pretrained:
        state_dict = None
        if pretrained_weight:
            state_dict = torch.load(pretrained_weight)
        elif arch in model_urls:
            if not os.path.exists(opt.pretrained_model_path):
                os.makedirs(opt.pretrained_model_path)
            state_dict = model_zoo.load_url(model_urls[arch], model_dir=opt.pretrained_model_path)
        else:
            assert False, f"model_urls not given {arch}."

        if state_dict:
            model_dict = model.state_dict()

            # 将3通道预训练权重转换为2通道
            conv1_3ch = state_dict['features.0.weight']
            new = torch.zeros(64, 1, 3, 3)
            for i, output_channel in enumerate(conv1_3ch):
                new[i] = 0.299 * output_channel[0] + 0.587 * output_channel[1] + 0.114 * output_channel[2]
            state_dict['features.0.weight'] = torch.cat((new, new), dim=1)

            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
            
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(f'loaded pretrained {arch} for depth from {pretrained_weight if pretrained_weight else model_urls[arch]}')
    
    return model

if __name__ == '__main__':

    if not os.path.exists(opt.pretrained_model_path):
        os.makedirs(opt.pretrained_model_path)

    opt.without_mask = False
    model_4ch = vggnet('vgg16', pretrained=False, num_classes=opt.class_num)
    input_4ch = torch.randn(1, 4, 224, 224)
    output_4ch = model_4ch(input_4ch)
    print(f'输入: {input_4ch.shape}, 输出: {output_4ch.shape}')
    print(f'第一层: {model_4ch.features[0]}')
    print("-" * 30)

    opt.without_mask = True
    model_3ch = vggnet('vgg16', pretrained=False, num_classes=opt.class_num)
    input_3ch = torch.randn(1, 3, 224, 224)
    output_3ch = model_3ch(input_3ch)
    print(f'输入: {input_3ch.shape}, 输出: {output_3ch.shape}')
    print(f'第一层: {model_3ch.features[0]}')
    print("-" * 30)
    
    model_depth = vggnet_for_depth('vgg16', pretrained=False, num_classes=opt.class_num)
    input_depth = torch.randn(1, 2, 224, 224)
    output_depth = model_depth(input_depth)
    print(f'深度图输入: {input_depth.shape}, 输出: {output_depth.shape}')
    print(f'第一层: {model_depth.features[0]}')
