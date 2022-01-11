import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np


# from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            # elif isinstance(m, SynchronizedBatchNorm2d):
            # m.weight.data.fill_(1)
            # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        elif output_stride == 4:
            dilations = [1, 3, 5, 7]
        else:
            raise NotImplementedError

        if inplanes == 1:
            planes = 9
        else:
            planes = inplanes

        self.inplanes = inplanes

        self.aspp1 = _ASPPModule(inplanes, planes, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, planes, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, planes, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, planes, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        idx = [1, 9, 25, 49].index(inplanes)

        c = [1, 9, 25, 49, 81]
        num = [1, 9 * 2, 25 * 3, 49 * 4]
        self.conv1 = nn.Conv2d(num[idx], c[idx + 1], 1, bias=False)
        self.bn1 = BatchNorm(c[idx + 1])
        self.relu = nn.ReLU(inplace=True)
        # @ self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        # pdb.set_trace()
        if self.inplanes == 1:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            return x

        if self.inplanes == 9:
            x1 = self.aspp1(x)
            x2 = self.aspp2(x)

            x = torch.cat((x1, x2), dim=1)

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            return x

        if self.inplanes == 25:
            x1 = self.aspp1(x)
            x2 = self.aspp2(x)
            x3 = self.aspp3(x)

            x = torch.cat((x1, x2, x3), dim=1)

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            return x

        if self.inplanes == 49:
            x1 = self.aspp1(x)
            x2 = self.aspp2(x)
            x3 = self.aspp3(x)
            x4 = self.aspp4(x)
            x = torch.cat((x1, x2, x3, x4), dim=1)

            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            return x

        return 0

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            # elif isinstance(m, SynchronizedBatchNorm2d):
            # m.weight.data.fill_(1)
            # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_aspp(backbone, output_stride, BatchNorm):
    return ASPP(backbone, output_stride, BatchNorm)
