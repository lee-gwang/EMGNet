import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os

class BinaryStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input>0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        zero_index = torch.abs(input) > 1
        middle_index = (torch.abs(input) <= 1) * (torch.abs(input) > 0.4)
        additional = 2-4*torch.abs(input)
        additional[zero_index] = 0.
        additional[middle_index] = 0.4
        return grad_input*additional

class MLConv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(MLConv2d, self).__init__()
        self.in_channels = in_c
        self.out_channels = out_c
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # upsample
        ## define weight
        self.weight = nn.Parameter(torch.Tensor(
            out_c, in_c // groups, kernel_size,kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_c))
        else:
            self.register_parameter('bias', None)
        self.th = 0
        self.threshold = nn.Parameter(self.th * torch.ones(1, out_c, 1, 1)) # version2, 1. pixel threshold --> channel sum --> sigmoid gate
        self.skeleton_mask = nn.Parameter(F.pad(torch.ones(3,3), (1, 1, 1, 1), 'constant', 0), requires_grad=False)

        self.step = BinaryStep.apply
        self.reset_parameters()
        self.num_out = 0
        self.num_full = 0

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        with torch.no_grad():
            self.threshold.data.fill_(0.)

    def forward(self, x):
        bs,c,h,w = x.shape # bs, channel, height, width
        skeleton_mask = self.skeleton_mask.view(1, 1, 5, 5)

        Y_3x3 = F.conv2d(x, self.weight * skeleton_mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
        Y_5x5 = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        mask = self.step( torch.sigmoid((Y_3x3.abs() - self.threshold).mean(dim=(2,3)))- 0.5).view(bs, c, 1, 1)


        self.num_out = mask.numel()
        self.num_full = mask[mask>0].numel()

        

        return Y_3x3 * (torch.ones_like(mask)-mask) + Y_5x5 *mask 
