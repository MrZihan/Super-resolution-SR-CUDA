#!/usr/bin/python3
# -*- coding:utf-8 -*-
import os.path
import sys
import getopt
import traceback
import numpy as np
from scipy.io import loadmat
import torch
from torch import nn
from collections import OrderedDict
import torch.nn.functional as F
import scipy
import struct

noise_level_model = 3  # noise level for model
sf = 2  # scale factor
x8 = False                           # default: False, x8 to boost performance
n_channels = 3            # fixed
nc = 128                  # fixed, number of channels
nb = 12                   # fixed, number of conv layers
model_pool = 'model_zoo'  # fixed
srmd_pca_file = os.path.join('kernels', 'srmd_pca_matlab.mat')
source = "."     # fixed
result = "."       # fixed
using_device = 'cpu'


def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)




def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR', negative_slope=0.2):
    L = []
    for t in mode:
        if t == 'C':
            L.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'v':
            L.append(nn.Upsample(scale_factor=4, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)

def upsample_pixelshuffle(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True, mode='2R', negative_slope=0.2):
    assert len(mode)<4 and mode[0] in ['2', '3', '4'], 'mode examples: 2, 2R, 2BR, 3, ..., 4BR.'
    up1 = conv(in_channels, out_channels * (int(mode[0]) ** 2), kernel_size, stride, padding, bias, mode='C'+mode, negative_slope=negative_slope)
    return up1

class SRMD(nn.Module):
    def __init__(self, in_nc=19, out_nc=3, nc=128, nb=12, upscale=4, act_mode='R'):
        """
        # ------------------------------------
        in_nc: channel number of input, default: 3+15
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        upscale: scale factor
        act_mode: batch norm + activation function; 'BR' means BN+ReLU
        upsample_mode: default 'pixelshuffle' = conv + pixelshuffle
        # ------------------------------------
        """
        super(SRMD, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True

        m_head = conv(in_nc, nc, mode='C'+act_mode[-1], bias=bias)
        m_body = [conv(nc, nc, mode='C'+act_mode, bias=bias) for _ in range(nb-2)]
        m_tail = upsample_pixelshuffle(nc, out_nc, mode=str(upscale), bias=bias)
        self.model = sequential(m_head, *m_body, m_tail)

def fspecial_gaussian(hsize, sigma):
    hsize = [hsize, hsize]
    siz = [(hsize[0]-1.0)/2.0, (hsize[1]-1.0)/2.0]
    std = sigma
    [x, y] = np.meshgrid(np.arange(-siz[1], siz[1]+1), np.arange(-siz[0], siz[0]+1))
    arg = -(x*x + y*y)/(2*std*std)
    h = np.exp(arg)
    h[h < scipy.finfo(float).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h = h/sumh
    return h

def main():
    model_path = source
    in_nc = 18 if noise_level_model==-1 else 19
    device = torch.device(using_device)

    # ----------------------------------------
    # load model
    # ----------------------------------------

    model = SRMD(in_nc=in_nc, out_nc=n_channels, nc=nc, nb=nb,
                upscale=sf, act_mode='R')
    print(model)
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()
    for _, v in model.named_parameters():
        v.requires_grad = False
    kernel = fspecial_gaussian(15, 0.01)
    P = loadmat(srmd_pca_file)['P']
    degradation_vector = np.dot(P, np.reshape(kernel, (-1), order="F"))
    degradation_vector=np.reshape(degradation_vector,(-1))
    degradation_vector=degradation_vector.astype('float32')

    
    fp=open(result,'wb')
    fp.write(struct.pack('<i',sf))
    
    for i in degradation_vector:
        fp.write(struct.pack('<f',i))
    
    for name in model.state_dict():
        floatvector=model.state_dict()[name].numpy().astype('float32')
        if floatvector.shape==(128,) or floatvector.shape==(3*sf**2,):
            for i in floatvector:
                fp.write(struct.pack('<f',i))
        else:
            for i in range(len(floatvector)):
                for j in range(len(floatvector[i])):
                    for k in range(len(floatvector[i][j])):
                         fp.write(struct.pack('<f',floatvector[i][j][k][0]))
                         fp.write(struct.pack('<f',floatvector[i][j][k][1]))
                         fp.write(struct.pack('<f',floatvector[i][j][k][2]))
    fp.close()

def showhelp():
    print("  -i input-file        PyTorch model file (default='model_zoo')")
    print("  -o output-file       SRMD_CUDA model file")
    print("  -p srmd-pca-file     srmd blur kernel pca data path (default='kernels/srmd_pca_matlab.mat')")
    print("  -s scale             upscale ratio (2/3/4, default=2)")
    print("  -n noise-level       denoise level (-1/0/1/2/3/4/5/6/7/8/9/10, default=3)")
    print("")
    print("Note:")
    print("")
    print("  1. input-file and output-file only accept file path (not directory path)")
    print("  2. This script uses the model trained by the original author for prediction. If necessary, please train the model, ")
    print("  define blur kernel and PCA dimension reduction data by yourself.")
    print("  3. scale = scale level, 2 = upscale 2x, 3 = upscale 3x, 4 = upscale 4x,please ensure the scale level corresponds to PyTorch model")
    print("  4. noise-level = noise level, larger value means stronger denoise effect, -1 = no effect")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        showhelp()
        sys.exit()
    try:
        opts, args = getopt.getopt(sys.argv[1:], "-i:-o:-p:-s:-n:", ["input-file", "output-file", "srmd-pca-file","scale","noise-level"])
    except getopt.GetoptError:
        print("Unrecognized Parameter exists.")
        showhelp()
        sys.exit()
    for o, a in opts:
        if o in ("-i", "--input-file"):
            source = a
        elif o in ("-o", "--output-file"):
            result = a
        elif o in ("-p", "--srmd-pca-file"):
            srmd_pca_file = a
        elif o in ("-s", "--scale"):
            try:
                sf = int(a)
            except Exception:
                print("Incorrect scale factor!")
                print("")
                print("upscale ratio (2/3/4, default=2)")
                sys.exit()
        elif o in ("-n", "--noise-level"):
            try:
                noise_level_model = int(a)
            except Exception:
                print("Incorrect noise level!")
                print("")
                print("denoise level (-1/0/1/2/3/4/5/6/7/8/9/10, default=3)")
                sys.exit()
    print(source)
    if not os.path.isfile(source):
        print("Input file doesn't exist!")
    elif not os.path.isfile(srmd_pca_file) or not os.path.splitext(srmd_pca_file)[1] == ".mat":
        print("Your SRMD PCA path doesn't exists or you didn't specify a Matlab Data file (.mat)!")
    elif sf > 4 or sf < 2:
        print("Incorrect scale factor!")
        print("")
        print("upscale ratio (2/3/4, default=2)")
    elif noise_level_model > 10 or noise_level_model < -1:
        print("Incorrect noise level!")
        print("")
        print("denoise level (-1/0/1/2/3/4/5/6/7/8/9/10, default=3)")
    else:
        main()
