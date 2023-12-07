import torch
from torch import nn

'''
This version:
In training: Add data augmentation of flip & rotation;
In testing: Add rotation 

Rotation: along the axial-direction (i.e., z-direction)
Flip: on x-y plane, fix z-direction
'''

'''
This version is based on 3D_2D_Jan01_v1.py 

(1) Follow the pipeline of DenseNet: conv(stride=2) + 3D-2D block + pooling + 3D-2D block
(2) 2 side paths as deep supervision
(3) Add auxiliary supervision for 2D branches in all scales 

(4) Replace 2D conv (e.g., 3*3) with 3D conv (3*3*1)
(5) Reduce the number of channels

(6) seg results of 2D branches are used as 2nd input for fusion bloack for ensemble purpose.
(7) Make fusion conv layers more complicated.

(8) Sparse aggregated nets to reduce the parameters and computational requirement.
(9) Use more layers to enlarge the receptive field. 


(1) split-and-stack are in the branch-level ( rather than in the block-level )
   In this v1, I use Sparse Dense Net in each branch
(2) 3D ensemble net is built opon the 2D branches. 

'''


imageType = 1 # grey-scale
ws = 96 # window size in x- and y-
ws_z = 40 # window size in z-

nclass = 3 # output
batch_size = 4

class_cost = [1.0, 5.0, 10.0] # for HVSMR'16 datasets 


class ConvBlock(nn.Module):
   def __init__(self, k1, k2, s1, s2, axis, nin, nout):
      super(ConvBlock, self).__init__()
      if axis == 0:
         if k2 == 3:
            self.conv = nn.Conv3d(nin, nout, (k1,k2,k2), stride=(s1,s2,s2), padding=(0, 1, 1))
         else:
            self.conv = nn.Conv3d(nin, nout, (k1,k2,k2), stride=(s1,s2,s2))
      elif axis == 1:
         if k2 == 3:
            self.conv = nn.Conv3d(nin, nout, (k2,k1,k2), stride=(s2,s1,s2), padding=(1, 0, 1))
         else:
            self.conv = nn.Conv3d(nin, nout, (k1,k2,k2), stride=(s1,s2,s2))
      elif axis == 2:
         if k2 == 3:
            self.conv = nn.Conv3d(nin, nout, (k2,k2,k1), stride=(s2,s2,s1), padding=(1, 1, 0))
         else:
            self.conv = nn.Conv3d(nin, nout, (k1,k2,k2), stride=(s1,s2,s2))
      else:
         if k2 == 3:
            self.conv = nn.Conv3d(nin, nout, (k2,k2,k2), stride=(s2,s2,s2), padding=(1, 1, 1))         
         else:
            self.conv = nn.Conv3d(nin, nout, (k1,k2,k2), stride=(s1,s2,s2))


   def forward(self, x):
      out = self.conv(x)
      return out

class SparseAggStack(nn.Module):
   def __init__(self, k1, k2, s1, s2, axis, nin):
      super(SparseAggStack, self).__init__()
      nout = 12
      self.block1 = nn.Sequential(
         ConvBlock(k1,k2,s1,s2,axis,nin,nout), 
         nn.BatchNorm3d(nout),
         nn.ReLU(inplace=True)
      )

      self.block2 = nn.Sequential(
         ConvBlock(k1,k2,s1,s2,axis,nin+nout,nout), 
         nn.BatchNorm3d(nout),
         nn.ReLU(inplace=True)
      )

      self.block3 = nn.Sequential(
         ConvBlock(k1,k2,s1,s2,axis,nout*2,nout), 
         nn.BatchNorm3d(nout),
         nn.ReLU(inplace=True)
      )

      self.block4 = nn.Sequential(
         ConvBlock(k1,k2,s1,s2,axis,nin+nout*2,nout), 
         nn.BatchNorm3d(nout),
         nn.ReLU(inplace=True)
      )

      self.block5 = nn.Sequential(
         ConvBlock(k1,k2,s1,s2,axis,nout*3,nout), 
         nn.BatchNorm3d(nout),
         nn.ReLU(inplace=True)
      )

      self.block6 = nn.Sequential(
         ConvBlock(k1,k2,s1,s2,axis,nout*3,nout), 
         nn.BatchNorm3d(nout),
         nn.ReLU(inplace=True)
      )

      self.block7 = nn.Sequential(
         ConvBlock(k1,k2,s1,s2,axis,nout*3,nout), 
         nn.BatchNorm3d(nout),
         nn.ReLU(inplace=True)
      )

      self.block8 = nn.Sequential(
         ConvBlock(k1,k2,s1,s2,axis,nin+nout*3,nout), 
         nn.BatchNorm3d(nout),
         nn.ReLU(inplace=True)
      )

      self.block9 = nn.Sequential(
         ConvBlock(k1,k2,s1,s2,axis,nout*4,nout), 
         nn.BatchNorm3d(nout),
         nn.ReLU(inplace=True)
      )

      self.block10 = nn.Sequential(
         ConvBlock(k1,k2,s1,s2,axis,nout*4,nout), 
         nn.BatchNorm3d(nout),
         nn.ReLU(inplace=True)
      )

      self.block11 = nn.Sequential(
         ConvBlock(k1,k2,s1,s2,axis,nout*4,nout), 
         nn.BatchNorm3d(nout),
         nn.ReLU(inplace=True)
      )

      self.block12 = nn.Sequential(
         ConvBlock(k1,k2,s1,s2,axis,nout*4,nout), 
         nn.BatchNorm3d(nout),
         nn.ReLU(inplace=True)
      )

   def forward(self, x):
      out1 = self.block1(x)
      out2 = self.block2(torch.cat([x, out1], dim=1))
      out3 = self.block3(torch.cat([out1, out2], dim=1))
      out4 = self.block4(torch.cat([x, out2, out3], dim=1))
      out5 = self.block5(torch.cat([out1, out3, out4], dim=1))
      out6 = self.block6(torch.cat([out2, out4, out5], dim=1))
      out7 = self.block7(torch.cat([out3, out5, out6], dim=1))
      out8 = self.block8(torch.cat([x, out4, out6, out7], dim=1))
      out9 = self.block9(torch.cat([out1, out5, out7, out8], dim=1))
      out10 = self.block10(torch.cat([out2,out6, out8, out9], dim=1))
      out11 = self.block11(torch.cat([out3,out7, out9, out10], dim=1))
      out12 = self.block12(torch.cat([out4,out8, out10, out11], dim=1))

      out = torch.cat([x, out5, out9, out11, out12], dim=1)
      return out


class DeConv(nn.Module):
   def __init__(self, nin, nout, axis):
      super(DeConv, self).__init__()
      if axis == 0:
         self.deconv = nn.ConvTranspose3d(nin, nout, kernel_size=(1, 2, 2), stride=(1, 2, 2))
      elif axis == 1:
         self.deconv = nn.ConvTranspose3d(nin, nout, kernel_size=(2, 1, 2), stride=(2, 1, 2))
      elif axis == 2:
         self.deconv = nn.ConvTranspose3d(nin, nout, kernel_size=(2, 2, 1), stride=(2, 2, 1))
      else:
         self.deconv = nn.ConvTranspose3d(nin, nout, kernel_size=(2, 2, 2), stride=(2, 2, 2))
   def forward(self, x):
      out = self.deconv(x)
      return out


class MaxPooling(nn.Module):
   def __init__(self, axis):
      super(MaxPooling, self).__init__()
      if axis == 0:
         self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2),stride=(1, 2, 2))
      elif axis == 1:
         self.pool = nn.MaxPool3d(kernel_size=(2, 1, 2),stride=(2, 1, 2))
      elif axis == 2:
         self.pool = nn.MaxPool3d(kernel_size=(2, 2, 1),stride=(2, 2, 1))
      else:
         self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2),stride=(2, 2, 2))
   def forward(self, x):
      out = self.pool(x)
      return out


class SparseAggBlock(nn.Module):
   def __init__(self, axis, nc):
      super(SparseAggBlock, self).__init__()
      self.conv1 = ConvBlock(1, 3, 1, 2, axis, 1, nc)
      self.conv2 = ConvBlock(1, 1, 1, 1, axis, 64, 3)
      self.conv3 = ConvBlock(1, 1, 1, 1, axis, 56, 3)

      self.aggregate1 = SparseAggStack(1, 3, 1, 1, axis, nc)
      self.aggregate2 = SparseAggStack(1, 3, 1, 1, axis, 64)

      self.conv_block1 = nn.Sequential(
         ConvBlock(1,1,1,1,axis,nin=64,nout=64), 
         nn.BatchNorm3d(64),
         nn.ReLU(inplace=True)
      )
      self.conv_block2 = nn.Sequential(
         ConvBlock(1,1,1,1,axis,nin=112,nout=112), 
         nn.BatchNorm3d(112),
         nn.ReLU(inplace=True)
      )

      self.maxpool = MaxPooling(axis)
      
      self.deconv1 = DeConv(3, 3, axis)
      self.deconv2 = DeConv(112, 112, axis)
      self.deconv3 = DeConv(112, 56, axis)
      
   def forward(self, x):
      S1_conv = self.conv1(x)
      S2= self.aggregate1(S1_conv)
      S2_conv = self.conv_block1(S2)
      S2_pool = self.maxpool(S2_conv)
      S3 = self.aggregate2(S2_pool)
      up1_conv = self.conv2(S2_conv)
      up1 = self.deconv1(up1_conv)
      S3_conv = self.conv_block2(S3)
      up2_1 = self.deconv2(S3_conv)
      up2_2 = self.deconv3(up2_1)
      up2 = self.conv3(up2_2)      
      return up1, up2

class HFA(nn.Module):
   def __init__(self, nin, nc=16):
      super(HFA, self).__init__()
      self.sparse_block1 = SparseAggBlock(0, nc)
      self.sparse_block2 = SparseAggBlock(1, nc)
      self.sparse_block3 = SparseAggBlock(2, nc)

      self.sparse_stack1 = SparseAggStack(3, 3, 1, 1, 3, nc)
      self.sparse_stack2 = SparseAggStack(3, 3, 1, 1, 3, 128)

      self.conv1 = nn.Conv3d(4, nc, 3, 2, 1)
      self.conv2 = nn.Conv3d(128, 128, 1, 1)
      self.conv3 = nn.Conv3d(128, 3, 1, 1)
      self.conv4 = nn.Conv3d(176, 176, 1, 1)
      self.conv5 = nn.Conv3d(64, 3, 1, 1)

      self.conv_block1 = nn.Sequential(
         nn.Conv3d(128, 128, 1, 1),
         nn.BatchNorm3d(128),
         nn.ReLU(inplace=True)
      )
      self.maxpool = nn.MaxPool3d(2, 2)

      self.deconv1 = nn.ConvTranspose3d(3, 3, 2, 2)
      self.deconv2 = nn.ConvTranspose3d(176, 128, 2, 2)
      self.deconv3 = nn.ConvTranspose3d(128, 64, 2, 2)

   def forward(self, x):
      xy_up1, xy_up2 = self.sparse_block1(x)
      xz_up1, xz_up2 = self.sparse_block2(x)
      yz_up1, yz_up2 = self.sparse_block3(x)

      ref_up1 = xy_up1 + xz_up1 + yz_up1
      ref_ens1 = torch.cat([ref_up1, x], dim=1)
      ref_ens1_conv = self.conv1(ref_ens1)
      ens1_s2 = self.sparse_stack1(ref_ens1_conv)
      ref_up2 = xy_up2 + xz_up2 + yz_up2
      ref_ens2 = torch.cat([ref_up2, x], dim=1)
      ref_ens2_conv = self.conv1(ref_ens2)
      ens2_s2 = self.sparse_stack1(ref_ens2_conv)
      ens_s2 = torch.cat([ens1_s2, ens2_s2], dim=1)
      
      ETc1_conv = self.conv2(ens_s2)
      ETc2_conv = self.conv_block1(ETc1_conv)
      ETc2_pool = self.maxpool(ETc2_conv)
      ETc3 = self.sparse_stack2(ETc2_pool)
      ETup1_conv = self.conv3(ETc2_conv)
      ETup1 = self.deconv1(ETup1_conv)
      ETc3_conv = self.conv4(ETc3)
      ETup2_1 = self.deconv2(ETc3_conv)
      ETup2_2 = self.deconv3(ETup2_1)
      ETup2 = self.conv5(ETup2_2)
      return ETup2
