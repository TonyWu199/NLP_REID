import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from collections import namedtuple
import numpy as np

from .resnet import BasicBlock, Bottleneck, remove_fc, resnet50
MODEL_REGISTRY = {}

# def build_textual_model(model_name, *args, **kwargs):
#     print('Getting registed visual model {}'.format(model_name))
#     return MODEL_REGISTRY[model_name](*args, **kwargs)

# def register_model(name):
#     '''DEcorator and register a new model'''
#     def register_model_cls(cls):
#         if name in MODEL_REGISTRY:
#             raise ValueError('Cannot register duplicate model ({})'.format(name))
#         MODEL_REGISTRY[name] = cls
#         return cls
#     return register_model_cls

class ResNet(nn.Module):
    def __init__(self, model_arch, res5_stride=2, res5_dilation=1, use_c4=False, pretrained=True, num_stripes=0):
        super(ResNet, self).__init__()
        block = model_arch.block
        layers = model_arch.stage

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.out_channels = 512 * block.expansion


        # stripes
        self.num_stripes = num_stripes
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # !attention, 这俩要添加self.inplanes的设置
        if self.num_stripes != 0:
            self.inplanes = 512
            self.layer3_part = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4_part = self._make_layer(block, 512, layers[3], stride=1)


        pretrained = True
        if pretrained:
            self.load_state_dict(
                remove_fc(model_zoo.load_url(model_arch.url)), strict=False
            )
        else:
            self._init_weight()

        self.use_c4 = use_c4
        if self.use_c4:
            del self.layer4
            self.out_channels = 256 * block.expansion

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        
        # branch global
        x_global = self.layer3(x)
        x_global = self.layer4(x_global)


        if self.num_stripes != 0:
            # branch local
            # x_local = self.layer3_part(x)
            # x_local = self.layer4_part(x_local)
            stripe_h = int(x_local.size(2) / self.num_stripes)
            local_feature_list = [] #torch.zeros(x.size(0), self.num_stripes, x.size(-1))
            for i in range(self.num_stripes):
                local_x = self.avg_pool(x_local[:,:, i*stripe_h: (i+1)*stripe_h, :]).squeeze()
                local_feature_list.append(local_x)
            return x_global, local_feature_list
            # return x_global, x_global


        return x_global

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

resnet = namedtuple('resnet', ['block', 'stage', 'url'])
model_archs = {}
model_archs['resnet18'] = resnet(BasicBlock, [2, 2, 2, 2],
                                 'https://download.pytorch.org/models/resnet18-5c106cde.pth')
model_archs['resnet34'] = resnet(BasicBlock, [3, 4, 6, 3],
                                 'https://download.pytorch.org/models/resnet34-333f7ec4.pth')
model_archs['resnet50'] = resnet(Bottleneck, [3, 4, 6, 3],
                                 'https://download.pytorch.org/models/resnet50-19c8e357.pth')
model_archs['resnet101'] = resnet(Bottleneck, [3, 4, 23, 3],
                                  'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
model_archs['resnet152'] = resnet(Bottleneck, [3, 8, 36, 3],
                                  'https://download.pytorch.org/models/resnet152-b121ed2d.pth')

# -----------------
# PCB Model
# -----------------
# from torchvision.models import resnet50
import torch.nn.init as init
class PCBresnet(nn.Module):
    def __init__(self, last_conv_stride=1, last_conv_dilation=1, num_stripes=6, out_channels=256, num_classes=11003):
        super(PCBresnet, self).__init__()
        self.base = resnet50(
            pretrained=True,
            last_conv_stride=last_conv_stride,
            last_conv_dilation=last_conv_dilation)
        self.num_stripes = num_stripes

        self.local_conv_list = nn.ModuleList()
        for strip in range(num_stripes):
            self.local_conv_list.append(nn.Sequential(
                nn.Conv2d(2048, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True) 
            ))
        
        if num_classes > 0:
            self.fc_list = nn.ModuleList()
            for stripe in range(num_stripes):
                fc = nn.Linear(out_channels, num_classes)
                init.normal_(fc.weight, std=0.001)
                init.constant_(fc.bias, 0)
                self.fc_list.append(fc) 

        self.out_channels = out_channels * self.num_stripes

    def forward(self, x):
        # shape [N, C, H, W]
        feat = self.base(x)
        assert feat.size(2) % self.num_stripes == 0 
        stripe_h = int(feat.size(2) / self.num_stripes)
        local_feat_list = []
        logits_list = []
        for i in range(self.num_stripes):
            local_feat = F.avg_pool2d(
                feat[:,:, i*stripe_h: (i+1) * stripe_h, :],
                (stripe_h, feat.size(-1))
            )
            local_feat = self.local_conv_list[i](local_feat)
            local_feat = local_feat.view(local_feat.size(0), -1)
            local_feat_list.append(local_feat)
            if hasattr(self, 'fc_list'):
                logits_list.append(self.fc_list[i](local_feat))

        global_feat = torch.cat(local_feat_list, dim=1)

        if hasattr(self, 'fc_list'):
            return global_feat, logits_list
        return global_feat

def build_resnet(cfg):
    print('Getting registed visual model {}'.format(cfg.MODEL.VISUAL_MODEL.NAME))
    if 'resnet' in cfg.MODEL.VISUAL_MODEL.NAME[:6]:
        model_arch = model_archs[cfg.MODEL.VISUAL_MODEL.NAME]
        model = ResNet(model_arch,
                        cfg.MODEL.VISUAL_MODEL.RES5_STRIDE,
                        cfg.MODEL.VISUAL_MODEL.RES5_DILATION,
                        use_c4=cfg.MODEL.VISUAL_MODEL.USE_C4,
                        pretrained=True,
                        num_stripes = cfg.MODEL.VISUAL_MODEL.NUM_STRIPES)
                        
    elif 'PCB' in cfg.MODEL.VISUAL_MODEL.NAME:
        model = PCBresnet(
            last_conv_stride=1, 
            last_conv_dilation=1, 
            num_stripes=6, 
            out_channels=256, 
            num_classes=11003
        )

    return model
  
if __name__ == '__main__':
    resnet = build_resnet()
    print(resnet)