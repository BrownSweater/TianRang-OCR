#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/6/14 12:33
# @Author : jj.wang



import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['MobileNetV3', 'mobilenetv3', 'mobilenetv3_small', 'mobilenetv3_large']


def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()
        self.max_stide = max(stride) if isinstance(stride, (list, tuple)) else stride
        assert self.max_stide in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = self.max_stide == 1 and inp == oup

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == 'RE':
            nlin_layer = nn.ReLU # or ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # dw
            conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, n_class=1000, input_size=224, dropout=0.8, mode='small', scale=1.0, algorithm='det', **kwargs):
        super(MobileNetV3, self).__init__()
        if algorithm == 'rec':
            stride = (2, 1)
        elif algorithm == 'det':
            stride = 2
        else:
            raise (f'{algorithm} must be det or rec')
        input_channel = 16
        last_channel = 1280
        if mode == 'large':
            # refer to Table 1 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  False, 'RE', 1],
                [3, 64,  24,  False, 'RE', 2],
                [3, 72,  24,  False, 'RE', 1],
                [5, 72,  40,  True,  'RE', stride],
                [5, 120, 40,  True,  'RE', 1],
                [5, 120, 40,  True,  'RE', 1],
                [3, 240, 80,  False, 'HS', stride],
                [3, 200, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 184, 80,  False, 'HS', 1],
                [3, 480, 112, True,  'HS', 1],
                [3, 672, 112, True,  'HS', 1],
                [5, 672, 160, True,  'HS', stride],
                [5, 960, 160, True,  'HS', 1],
                [5, 960, 160, True,  'HS', 1],
            ]
        elif mode == 'small':
            # refer to Table 2 in paper
            mobile_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  True,  'RE', 2],
                [3, 72,  24,  False, 'RE', stride],
                [3, 88,  24,  False, 'RE', 1],
                [5, 96,  40,  True,  'HS', stride],
                [5, 240, 40,  True,  'HS', 1],
                [5, 240, 40,  True,  'HS', 1],
                [5, 120, 48,  True,  'HS', 1],
                [5, 144, 48,  True,  'HS', 1],
                [5, 288, 96,  True,  'HS', stride],
                [5, 576, 96,  True,  'HS', 1],
                [5, 576, 96,  True,  'HS', 1],
            ]
        else:
            raise NotImplementedError

        # building first layer
        assert input_size % 32 == 0
        last_channel = make_divisible(last_channel * scale) if scale > 1.0 else last_channel
        self.conv1 = [conv_bn(3, input_channel, 2, nlin_layer=Hswish)]
        self.classifier = []

        stage_names = ['stage{}'.format(i) for i in [1, 2, 3, 4]]
        stage_index = 0
        seq = []
        # building mobile blocks
        for index, (k, exp, c, se, nl, s) in enumerate(mobile_setting):
            if isinstance(s, int) and s==1:
                output_channel = make_divisible(c * scale)
                exp_channel = make_divisible(exp * scale)
                self.conv1.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            else:
                break
        self.conv1 = nn.Sequential(*self.conv1)
        for k, exp, c, se, nl, s in mobile_setting[index:]:
            output_channel = make_divisible(c * scale)
            exp_channel = make_divisible(exp * scale)
            if (isinstance(s, int) and s==2) or isinstance(s, (tuple, set)) and 2 in s:
                if len(seq) > 0:
                    setattr(self, stage_names[stage_index], nn.Sequential(*seq))
                    stage_index += 1
                seq = []
            seq.append(MobileBottleneck(input_channel, output_channel, k, s, exp_channel, se, nl))
            input_channel = output_channel
            self.output_channel = output_channel
        setattr(self, stage_names[stage_index], nn.Sequential(*seq))
        # building last several layers
        # if mode == 'large':
        #     last_conv = make_divisible(960 * scale)
        #     seq.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
        #     setattr(self, stage_names[stage_index], nn.Sequential(*seq))
        #     # self.features.append(nn.AdaptiveAvgPool2d(1))
        #     # self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
        #     # self.features.append(Hswish(inplace=True))
        # elif mode == 'small':
        #     last_conv = make_divisible(576 * scale)
        #     seq.append(conv_1x1_bn(input_channel, last_conv, nlin_layer=Hswish))
        #     setattr(self, stage_names[stage_index], nn.Sequential(*seq))
        #     # self.features.append(SEModule(last_conv))  # refer to paper Table2, but I think this is a mistake
        #     # self.features.append(nn.AdaptiveAvgPool2d(1))
        #     # self.features.append(nn.Conv2d(last_conv, last_channel, 1, 1, 0))
        #     # self.features.append(Hswish(inplace=True))
        # else:
        #     raise NotImplementedError
        # self.output_channel = last_conv
        # make it nn.Sequential
        # self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),    # refer to paper section 6
            nn.Linear(last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        c2 = self.stage1(x)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        # x = x.mean(3).mean(2)
        # x = self.classifier(x)
        return c2, c3, c4, c5

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def mobilenetv3(pretrained=False, **kwargs):
    model = MobileNetV3(**kwargs)
    if pretrained:
        state_dict = torch.load('mobilenetv3_small_67.4.pth.tar')
        model.load_state_dict(state_dict, strict=False)
        # raise NotImplementedError
    return model

def mobilenetv3_small(pretrained=False, **kwargs):
    model = MobileNetV3(**kwargs)
    if pretrained:
        state_dict = torch.load('mobilenetv3_small_67.4.pth.tar')
        model.load_state_dict(state_dict, strict=False)
        # raise NotImplementedError
    return model

def mobilenetv3_large(pretrained=False, **kwargs):
    model = MobileNetV3(**kwargs)
    if pretrained:
        state_dict = torch.load('mobilenetv3_small_67.4.pth.tar')
        model.load_state_dict(state_dict, strict=False)
        pass
        # raise NotImplementedError
    return model

if __name__ == '__main__':
    net = mobilenetv3(algorithm='rec', mode='small', scale=0.5)
    print('mobilenetv3:\n', net)
    print('Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0))
    input_size=(1, 3, 32, 100)
    # pip install --upgrade git+https://github.com/kuan-wang/pytorch-OpCounter.git
    from thop import profile
    flops, params = profile(net, input_size=input_size)
    # print(flops)
    # print(params)
    print('Total params: %.2fM' % (params/1000000.0))
    print('Total flops: %.2fM' % (flops/1000000.0))
    x = torch.randn(input_size)
    out = net(x)
    for i in out:
        print(i.shape)
        print(type(i.shape[2]))