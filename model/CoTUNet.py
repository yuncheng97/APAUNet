import torch
from torch import nn
from torch import Tensor
import numpy as np
from typing import Union, Tuple, Optional
import torch.nn.functional as F


class Involution3d(nn.Module):
    """
    This class implements the 3d involution.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 sigma_mapping: Optional[nn.Module] = None,
                 kernel_size: Union[int, Tuple[int, int, int]] = (1, 1, 1),
                 stride: Union[int, Tuple[int, int, int]] = (1, 1, 1),
                 groups: int = 1,
                 reduce_ratio: int = 1,
                 dilation: Union[int, Tuple[int, int, int]] = (1, 1, 1),
                 padding: Union[int, Tuple[int, int, int]] = (0, 0, 0),
                 bias: bool = False,
                 **kwargs) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param sigma_mapping: (nn.Module) Non-linear mapping as introduced in the paper. If none BN + ReLU is utilized
        :param kernel_size: (Union[int, Tuple[int, int, int]]) Kernel size to be used
        :param stride: (Union[int, Tuple[int, int, int]]) Stride factor to be utilized
        :param groups: (int) Number of groups to be employed
        :param reduce_ratio: (int) Reduce ration of involution channels
        :param dilation: (Union[int, Tuple[int, int, int]]) Dilation in unfold to be employed
        :param padding: (Union[int, Tuple[int, int, int]]) Padding to be used in unfold operation
        :param bias: (bool) If true bias is utilized in each convolution layer
        :param **kwargs: Unused additional key word arguments
        """
        # Call super constructor
        super(Involution3d, self).__init__()
        # Check parameters
        assert isinstance(in_channels, int) and in_channels > 0, "in channels must be a positive integer."
        assert in_channels % groups == 0, "out_channels must be divisible by groups"
        assert isinstance(out_channels, int) and out_channels > 0, "out channels must be a positive integer."
        assert out_channels % groups == 0, "out_channels must be divisible by groups"
        assert isinstance(sigma_mapping, nn.Module) or sigma_mapping is None, \
            "Sigma mapping must be an nn.Module or None to utilize the default mapping (BN + ReLU)."
        assert isinstance(kernel_size, int) or isinstance(kernel_size, tuple), \
            "kernel size must be an int or a tuple of ints."
        assert isinstance(stride, int) or isinstance(stride, tuple), \
            "stride must be an int or a tuple of ints."
        assert isinstance(groups, int), "groups must be a positive integer."
        assert isinstance(reduce_ratio, int) and reduce_ratio > 0, "reduce ratio must be a positive integer."
        assert isinstance(dilation, int) or isinstance(dilation, tuple), \
            "dilation must be an int or a tuple of ints."
        assert isinstance(padding, int) or isinstance(padding, tuple), \
            "padding must be an int or a tuple of ints."
        assert isinstance(bias, bool), "bias must be a bool"
        # Save parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.groups = groups
        self.reduce_ratio = reduce_ratio
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        self.bias = bias
        # Init modules
        self.sigma_mapping = sigma_mapping if sigma_mapping is not None else nn.Sequential(
            nn.BatchNorm3d(num_features=self.out_channels // self.reduce_ratio, momentum=0.3), nn.ReLU())
        self.initial_mapping = nn.Conv3d(
            in_channels=self.in_channels, out_channels=self.out_channels,
            kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0),
            bias=bias) if self.in_channels != self.out_channels else nn.Identity()
        self.o_mapping = nn.AvgPool3d(kernel_size=self.stride, stride=self.stride)
        self.reduce_mapping = nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=self.out_channels // self.reduce_ratio, kernel_size=(1, 1, 1),
            stride=(1, 1, 1), padding=(0, 0, 0), bias=bias)
        self.span_mapping = nn.Conv3d(
            in_channels=self.out_channels // self.reduce_ratio,
            out_channels=self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * self.groups,
            kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=bias)
        self.pad = nn.ConstantPad3d(padding=(self.padding[0], self.padding[0],
                                             self.padding[1], self.padding[1],
                                             self.padding[2], self.padding[2]), value=0.)

    def __repr__(self) -> str:
        """
        Method returns information about the module
        :return: (str) Info string
        """
        return ("{}({}, {}, kernel_size=({}, {}, {}), stride=({}, {}, {}), padding=({}, {}, {}), "
                "groups={}, reduce_ratio={}, dilation=({}, {}, {}), bias={}, sigma_mapping={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.kernel_size[0],
            self.kernel_size[1],
            self.kernel_size[2],
            self.stride[0],
            self.stride[1],
            self.stride[2],
            self.padding[0],
            self.padding[1],
            self.padding[2],
            self.groups,
            self.reduce_mapping,
            self.dilation[0],
            self.dilation[1],
            self.dilation[2],
            self.bias,
            str(self.sigma_mapping)
        ))

    def forward(self, input: torch.Tensor, weight:torch.Tensor) -> torch.Tensor:
    # def forward(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of the shape [batch size, in channels, depth, height, width]
               weight: (torch.Tensor) Input tensor of the shape [batch size, in channels, depth, height, width]
        :return: (torch.Tensor) Output tensor of the shape [batch size, out channels, depth, height, width] (w/ same padding)
        """
        # Check input dimension of input tensor
        assert input.ndimension() == 5, \
            "Input tensor to involution must be 5d but {}d tensor is given".format(input.ndimension())
        # Save input shapes and compute output shapes
        batch_size, _, in_depth, in_height, in_width = input.shape
        out_depth = (in_depth + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) \
                    // self.stride[0] + 1
        out_height = (in_height + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) \
                     // self.stride[1] + 1
        out_width = (in_width + 2 * self.padding[2] - self.dilation[2] * (self.kernel_size[2] - 1) - 1) \
                    // self.stride[2] + 1
        # Unfold and reshape input tensor
        input_initial = self.initial_mapping(input)
        input_unfolded = self.pad(input_initial) \
            .unfold(dimension=2, size=self.kernel_size[0], step=self.stride[0]) \
            .unfold(dimension=3, size=self.kernel_size[1], step=self.stride[1]) \
            .unfold(dimension=4, size=self.kernel_size[2], step=self.stride[2])
        input_unfolded = input_unfolded.reshape(batch_size, self.groups, self.out_channels // self.groups,
                                                self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2], -1)
        input_unfolded = input_unfolded.reshape(tuple(input_unfolded.shape[:-1])
                                                + (out_depth, out_height, out_width))
        # Generate kernel
        kernel = weight
        kernel = kernel.view(
            batch_size, self.groups, weight.shape[1],
            kernel.shape[-3], kernel.shape[-2], kernel.shape[-1]).unsqueeze(dim=2)
        output = (kernel * input_unfolded).sum(dim=3)
        # Reshape output
        output = output.view(batch_size, -1, output.shape[-3], output.shape[-2], output.shape[-1])
        return output


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or \
                isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=1e-2)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)
                
def swish(x, inplace: bool = False):
    """Swish - Described in: https://arxiv.org/abs/1710.05941
    """
    return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())


class Swish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return swish(x, self.inplace)

   
class CotLayer(nn.Module):
    def __init__(self, dim, kernel_size, project_dim=2):
        super(CotLayer, self).__init__()

        self.project_dim = project_dim
        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv3d(dim, dim, self.kernel_size, stride=1, padding=self.kernel_size//2, groups=4, bias=False),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True)
        )
    
        share_planes = 8
        factor = 2
        self.embed = nn.Sequential(
            nn.Conv3d(2*dim, dim//factor, 1, bias=False),
            nn.BatchNorm3d(dim//factor),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim//factor, dim, kernel_size=1),
            nn.GroupNorm(num_groups=dim // share_planes, num_channels=dim)
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm3d(dim)
        )

        self.local_conv = Involution3d(dim, dim)
        self.bn = nn.BatchNorm3d(dim)
        self.act = Swish(inplace=True)

        reduction_factor = 4
        self.radix = 2
        attn_chs = max(dim * self.radix // reduction_factor, 32)
        self.se = nn.Sequential(
            nn.Conv3d(dim, attn_chs, 1),
            nn.BatchNorm3d(attn_chs),
            nn.ReLU(inplace=True),
            nn.Conv3d(attn_chs, self.radix*dim, 1)
        )

    def forward(self, x):
        k = self.key_embed(x)
        q = x
        qk = torch.cat([q, k], dim=1)
        b, c, qk_hh, qk_ww, qk_dd= qk.size()

        w = self.embed(qk)
        
        v = self.conv1x1(x)
        x = self.local_conv(x, w)
        v = self.bn(v)
        v = self.act(v)

        B, C, H, W, D = v.shape
        v = v.view(B, C, 1, H, W, D)
        x = x.view(B, C, 1, H, W, D)
        x = torch.cat([x, v], dim=2)

        x_gap = x.sum(dim=2)
        x_gap = x_gap.mean((2, 3, 4), keepdim=True)
        x_attn = self.se(x_gap)
        x_attn = x_attn.view(B, C, self.radix)
        x_attn = F.softmax(x_attn, dim=2)
        out = (x * x_attn.reshape((B, C, self.radix, 1, 1, 1))).sum(dim=2)
        
        return out.contiguous()


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x
        
class doubelconv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(doubelconv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x


class SpatialAttentionEncoder(nn.Module):
    def __init__(self, in_ch, out_ch, last_layer=False):
        super(SpatialAttentionEncoder, self).__init__()
        self.last_layer = last_layer
        self.Maxpool = nn.MaxPool3d(kernel_size=2,stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, 1),
            nn.BatchNorm3d(in_ch),
            nn.ReLU(inplace=True)
            )        
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
            )
        self.attn = CotLayer(dim=in_ch, kernel_size=3)
        
    def forward(self, input):  
        input = self.attn(input)
        input = self.conv2(input)
        return input

class CoTUNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CoTUNet, self).__init__()

        self.initializer = InitWeights_He(1e-2)
        self.pool = nn.MaxPool3d(2)
        self.stem =  doubelconv_block(in_ch, 64)
        self.encoder1 = SpatialAttentionEncoder(64, 64)
        self.encoder2 = SpatialAttentionEncoder(64, 128)
        self.encoder3 = SpatialAttentionEncoder(128, 256)
        self.encoder4 = SpatialAttentionEncoder(256, 512)
        self.encoder5 = SpatialAttentionEncoder(512, 1024)
        
        self.decoder1 = nn.ConvTranspose3d(1024, 512, 2, stride=2)
        self.conv1    = doubelconv_block(1024, 512)
        self.decoder2 = nn.ConvTranspose3d(512, 256, 2, stride=2)
        self.conv2    = doubelconv_block(512, 256)
        self.decoder3 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.conv3    = doubelconv_block(256, 128)
        self.decoder4 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.conv4    = doubelconv_block(128, 64)
        
        self.tail = nn.Conv3d(64, out_ch, 1)

        self.apply(self.initializer)

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.encoder1(c1)
        p2 = self.pool(c2)
        c3 = self.encoder2(p2)
        p3 = self.pool(c3)
        c4 = self.encoder3(p3)
        p4 = self.pool(c4)
        c5 = self.encoder4(p4) 
        p5 = self.pool(c5)
        c6 = self.encoder5(p5)
        up_6 = self.decoder1(c6)
        merge6 = torch.cat([up_6, c5], dim=1)
        c7 = self.conv1(merge6)

        up_7 = self.decoder2(c7)
        merge7 = torch.cat([up_7, c4], dim=1)
        c8 = self.conv2(merge7)

        up_8 = self.decoder3(c8)
        merge8 = torch.cat([up_8, c3], dim=1)
        c9 = self.conv3(merge8)

        up_9 = self.decoder4(c9)
        merge9 = torch.cat([up_9, c2], dim=1)
        c10 = self.conv4(merge9)

        out = self.tail(c10)

        return out