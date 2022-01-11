import torch
from torch import nn
from torch.nn import functional as F
import network.unet as net
import network.conv4d as conv4d
import random
import numpy as np
import cv2
import pdb
import yaml
import time

"""
main network.
Conv4Ds is a efficient implementation from https://github.com/gengshan-y/VCN
feature of shape [N,C,K,K,H,W] for Conv4D 
It can be replaced by normal con2d with feature of shape [N,C,H,W]
In my experiment, it works slightly better than conv2d 
"""

class ResultsErasing(object):

    def __init__(self, top_n=0.10):
        self.top_n = top_n

    def __call__(self, input, x, ratio):
        # input: shape [N, 2, H, W]
        # x:     shape [N, C, H, W] input image
        # ratio: shape[N,]
        # f_p: number of foreground pixels (f_p)
        
        N, C, H, W = input.shape
        for i_c in range(C):
        
            for n in range(N): # N-dimension index one by one
                f_p = int(ratio[n]*H*W)
                # pdb.set_trace()
                # print("{}, {}, {}".format(ratio[n], int(f_p*self.top_n), H*W))
                max_list = input[n, i_c, ...].flatten().topk(int(f_p*self.top_n))[0] # top max value
                max_list_min = max_list[-1]
                # pdb.set_trace()
                for c in range(x.shape[1]): # C-dimension index one by one
                    x[n, c,...][input[n, i_c,...]>max_list_min]=0    # input 肯定单channel，因为二分类
        # pdb.set_trace()
        return x

class Dual_net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        
        # [32, 16, 16, 16, 16]

        # load parameters
        self.net = net.UNet(input_channels=cfg["INPUT_CHANNEL"], nb_filter=[32, 64, 96, 128, 196])

        # build model

        self.resultsErasing = ResultsErasing(top_n=cfg["RCE_RATIO"])

        self.max_affinity = cfg["MAX_AFFINITY"]
        self.base = np.array([k for k in range(1, self.max_affinity + 1)][::2])  # 1,3,5,7, ...

        assert len(self.base) >= 2, "Affinity matrix number must be greater than 3."
        self.base_2 = self.base ** 2

        self.b1 = self.block(in_planes=32, out_planes=32, kernel_size=3, padding=1)
        self.b2 = self.block(in_planes=32, out_planes=1, kernel_size=1, padding=0)
        
        self.output = nn.Conv2d(32, 1, kernel_size=3, padding=1, bias=False)
        # f(.)
        self.d = self.make_d()

        # g(.)
        self.d_max = conv4d.Conv4Ds(inplanes=1)
        

        
    def block(self, in_planes, out_planes, kernel_size, padding, bias=False):
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, bias=bias),
                             nn.BatchNorm2d(out_planes),
                             nn.ReLU(inplace=True))

    def res_block(in_planes=1, out_planes=25, kernel_size=3, padding=1):
        volume = self.generate_affinity(input, 5, sim=False)
        similarity_volume = self.generate_affinity(input, 5)
        similarity_volume = self.block(similarity_volume)
        
    
    def make_d(self):
        aspp = []
        for i in range(len(self.base_2) - 1):
            aspp.append(conv4d.Conv2D(self.base_2[i], self.base_2[i+1]))

        return nn.Sequential(*aspp)

    def build_f(self, input):
        a_1 = self.generate_affinity(input, self.base[1])
        d_1 = self.d[0](input)
        x = a_1 + d_1
        for i, k in zip(self.base[2:], range(1, len(self.base) - 1)):
            a_i = self.generate_affinity(input, i)
            d_i = self.d[k](x)
            x = a_i + d_i
        return x
        
    def build_g(self, input, x):
        max_affinity = self.max_affinity
        a_max = self.generate_affinity(input, max_affinity, sim=False)

        b, c, h, w = x.shape
        x = x.view(b, 1, self.max_affinity, self.max_affinity, h, w)
        # using pseudo-4d conv can decrease the parameter.
        # also, you can use 2d conv for [b,c,h,w]
        kernel = self.d_max(x).view(b, c, h, w)
        out = a_max * kernel

        out = torch.sum(out, dim=1) / self.base_2[-1]
        out = out.unsqueeze(1)
        return out

    def construct_coff(self, kernel_size, n, c, h, w):
        K = kernel_size
        h_0 = np.array([[i] * K for i in range(K)]).flatten()
        h_1 = h + h_0
        w_0 = np.array([[i] for i in range(K)] * K).flatten()
        w_1 = w + w_0
        return h_0, h_1, w_0, w_1

    def generate_affinity(self, input, kernel_size, sim=True):
        """
        :param input: [n, c, h, w]
        :return: [n, 9*c, h, w]
        """
        K = kernel_size
        n, c, h, w = input.shape
        pad_pixel_number = K // 2
        input_pad = F.pad(input, [pad_pixel_number] * 4)

        # construct affinity matrix
        affinity = []
        h_0, h_1, w_0, w_1 = self.construct_coff(K, n, c, h, w)
        for i in range(K ** 2):
            affinity.append(input_pad[:, :, h_0[i]:h_1[i], w_0[i]:w_1[i]])
        affinity = torch.cat(affinity, dim=1)
        if sim:
            affinity = affinity * input  # calculate the similarity
        return affinity

    def build_fine2coarse(self, input):
        """
        x is an [N, 1, H, W] initial segmentation map.
        """
        x = self.build_f(input)
        x = self.build_g(input, x)

        return torch.cat([1 - x, x], dim=1)

    def transform1(self, img, x, ratio):
        img = self.resultsErasing(img, x, ratio)
        
        return img
        
    def aux_transfer(self, aux_f):
        aux_f = self.b1(aux_f)
        aux_f = self.b2(aux_f)
        # aux_f = self.b3(aux_f)
        return aux_f
        
    def forward(self, x, ratio):
        s = time.time()
        aux_f, x1 = self.net(x)
        x1 = self.output(x1)
        aux_f = self.aux_transfer(aux_f)
        x1 = self.build_fine2coarse(x1+aux_f)   
        ss = time.time()
        # print(ss-s)
        
        x2 = self.transform1(x1, x, ratio)
        # pdb.set_trace()
        # cv2.imwrite("erase.tiff", (x2.cpu().numpy()[0,...].transpose(1,2,0)*255)[...,::-1].astype(np.uint8))
        aux_f, x2 = self.net(x2)
        x2 = self.output(x2)
        aux_f = self.aux_transfer(aux_f)
        x2 = self.build_fine2coarse(x2+aux_f)


        return x1, x2#, ss-s
