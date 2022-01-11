import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class projfeat4d(torch.nn.Module):
    '''
    Turn 3d projection into 2d projection
    '''

    def __init__(self, in_planes, out_planes, stride, with_bn=True, groups=1):
        super(projfeat4d, self).__init__()
        self.with_bn = with_bn
        self.stride = stride
        self.conv1 = nn.Conv3d(in_planes, out_planes, 1, (stride, stride, 1), padding=0, bias=not with_bn,
                               groups=groups)
        self.bn = nn.BatchNorm3d(out_planes)

    def forward(self, x):
        b, c, u, v, h, w = x.size()
        x = self.conv1(x.view(b, c, u, v, h * w))
        if self.with_bn:
            x = self.bn(x)
        _, c, u, v, _ = x.shape
        x = x.view(b, c, u, v, h, w)
        return x


class sepConv4d(torch.nn.Module):
    '''
    Separable 4d convolution block as 2 3D convolutions
    '''

    def __init__(self, in_planes, out_planes, stride=1, dilation=1, with_bn=True, ksize=3, full=True):
        super(sepConv4d, self).__init__()
        bias = not with_bn
        self.isproj = False
        self.stride = stride
        expand = 1

        if with_bn:
            if in_planes != out_planes:
                self.isproj = True
                self.proj = nn.Sequential(nn.Conv2d(in_planes, out_planes, 1, bias=bias, padding=0),
                                          nn.BatchNorm2d(out_planes))
            if full:
                self.conv1 = nn.Sequential(
                    nn.Conv2d(in_planes * expand, in_planes, (ksize, ksize), stride=(self.stride, self.stride),
                              bias=bias, padding=(dilation, dilation), dilation=dilation),
                    nn.BatchNorm2d(in_planes))
            else:
                self.conv1 = nn.Sequential(
                    nn.Conv2d(in_planes * expand, in_planes, (ksize, ksize), stride=1, bias=bias,
                              padding=(dilation, dilation), dilation=dilation),
                    nn.BatchNorm2d(in_planes))
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_planes, in_planes * expand, (ksize, ksize), stride=(self.stride, self.stride),
                          bias=bias, padding=(ksize // 2, ksize // 2)),
                nn.BatchNorm2d(in_planes * expand))
        else:
            if in_planes != out_planes:
                self.isproj = True
                self.proj = nn.Conv2d(in_planes, out_planes, 1, bias=bias, padding=0)
            if full:
                self.conv1 = nn.Conv2d(in_planes * expand, in_planes, (ksize, ksize),
                                       stride=(self.stride, self.stride), bias=bias,
                                       padding=(dilation, dilation), dilation=dilation)
            else:
                self.conv1 = nn.Conv2d(in_planes * expand, in_planes, (ksize, ksize), stride=1, bias=bias,
                                       padding=(dilation, dilation), dilation=dilation)
            self.conv2 = nn.Conv2d(in_planes, in_planes * expand, (ksize, ksize),
                                   stride=(self.stride, self.stride), bias=bias, padding=(ksize // 2, ksize // 2))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, u, v, h, w = x.shape

        x = x.view(b, c, u, v, -1).permute(0, 4, 1, 2, 3).contiguous().view(-1, c, u, v)
        x = self.conv2(x).view(b, -1, c, u, v).permute(0, 2, 3, 4, 1).contiguous()
        b, c, u, v, _ = x.shape
        x = self.relu(x)

        x = x.view(b, c, -1, h, w).permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)
        x = self.conv1(x).view(b, -1, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
        b, c, _, h, w = x.shape

        if self.isproj:
            x = self.proj(x.view(b, c, -1, w))
        x = x.view(b, -1, u, v, h, w)
        return x


class ASPP4D(torch.nn.Module):
    def __init__(self, inplanes, output_stride, BN=True):
        super(ASPP4D, self).__init__()

        if output_stride == 4:
            dilations = [1, 3, 5, 7]
        else:
            raise NotImplementedError

        self.aspp1 = nn.Sequential(sepConv4d(inplanes, inplanes, ksize=3, stride=1, dilation=dilations[0], with_bn=BN),
                                   nn.ReLU(inplace=True))

        self.aspp2 = nn.Sequential(sepConv4d(inplanes, inplanes, ksize=3, stride=1, dilation=dilations[1], with_bn=BN),
                                   nn.ReLU(inplace=True))

        self.aspp3 = nn.Sequential(sepConv4d(inplanes, inplanes, ksize=3, stride=1, dilation=dilations[2], with_bn=BN),
                                   nn.ReLU(inplace=True))

        self.aspp4 = nn.Sequential(sepConv4d(inplanes, inplanes, ksize=3, stride=1, dilation=dilations[3], with_bn=BN),
                                   nn.ReLU(inplace=True))

        self.conv = nn.Sequential(sepConv4d(4 * inplanes, inplanes, ksize=3, stride=1),
                                  nn.ReLU(inplace=True))

        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv(x)


        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Dilation4D(torch.nn.Module):
    def __init__(self, inplanes, BN=True):
        super(Dilation4D, self).__init__()

        self.aspp1 = nn.Sequential(sepConv4d(inplanes, inplanes, ksize=3, stride=1, dilation=1, with_bn=BN),
                                   nn.ReLU(inplace=True))
        self.aspp2 = nn.Sequential(sepConv4d(inplanes, inplanes, ksize=3, stride=1, dilation=1, with_bn=BN),
                                   nn.ReLU(inplace=True))
        self.aspp3 = nn.Sequential(sepConv4d(inplanes, inplanes, ksize=3, stride=1, dilation=1, with_bn=BN),
                                   nn.ReLU(inplace=True))
        self.aspp4 = nn.Sequential(sepConv4d(inplanes, inplanes, ksize=3, stride=1, dilation=7, with_bn=BN),
                                   nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.aspp1(x)
        x = self.aspp2(x)
        x = self.aspp3(x)
        x = self.aspp4(x)

        return x


class Conv2D(torch.nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Conv2D, self).__init__()

        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class Conv2Ds(torch.nn.Module):
    def __init__(self, inplanes):
        super(Conv2Ds, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes * 2, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(inplanes * 2)

        self.conv2 = nn.Conv2d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(inplanes * 2)

        self.conv3 = nn.Conv2d(inplanes * 2, inplanes, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(inplanes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        return x


class Conv4Ds(torch.nn.Module):
    def __init__(self, inplanes, BN=True):
        super(Conv4Ds, self).__init__()

        self.conv1 = nn.Sequential(sepConv4d(inplanes, inplanes * 2, ksize=3, stride=1, dilation=1, with_bn=BN),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(sepConv4d(inplanes * 2, inplanes * 2, ksize=3, stride=1, dilation=1, with_bn=BN),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(sepConv4d(inplanes * 2, inplanes, ksize=3, stride=1, dilation=1, with_bn=BN),
                                   nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x
