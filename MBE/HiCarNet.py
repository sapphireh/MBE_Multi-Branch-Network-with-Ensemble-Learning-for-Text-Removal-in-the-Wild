"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision import models
import numpy as np
import cv2
import copy
from PIL import Image
import torch.nn.functional as F
from pdb import set_trace as stx


##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)


##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )


    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA   = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x, is_unet=False):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res

class CAB2(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, reduction, bias, act):
        super(CAB2, self).__init__()
        modules_body = []
        modules_body.append(conv(in_channel, in_channel, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(in_channel, out_channel, kernel_size, bias=bias))

        self.CA   = CALayer(out_channel, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x   = self.body(x)
        res = self.CA(x)
        res += x
        return res
##########################################################################
## Patch Attention Block (PAB)
class PAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, scale_unetfeats, bias, act):
        super(PAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        self.body = nn.Sequential(*modules_body)

        modules_body2 = []
        modules_body2.append(conv(n_feat+scale_unetfeats, n_feat+scale_unetfeats, kernel_size, bias=bias))
        modules_body2.append(act)
        modules_body2.append(conv(n_feat+scale_unetfeats, n_feat+scale_unetfeats, kernel_size, bias=bias))
        self.body2 = nn.Sequential(*modules_body2)

        modules_body3 = []
        modules_body3.append(conv(n_feat+2*scale_unetfeats, n_feat+2*scale_unetfeats, kernel_size, bias=bias))
        modules_body3.append(act)
        modules_body3.append(conv(n_feat+2*scale_unetfeats, n_feat+2*scale_unetfeats, kernel_size, bias=bias))
        self.body3 = nn.Sequential(*modules_body3)

        self.conv1 = nn.Sequential(
            nn.Conv2d(n_feat, n_feat , 1, padding=0, bias=bias),
            nn.Sigmoid()
        )
        self.channeld1 = conv(n_feat*2, n_feat, kernel_size, bias=bias)

        self.conv2 = nn.Sequential(
            nn.Conv2d(n_feat+scale_unetfeats, n_feat+scale_unetfeats , 1, padding=0, bias=bias),
            nn.Sigmoid()
        )
        self.channeld2 = conv((n_feat+scale_unetfeats)*2, n_feat+scale_unetfeats, kernel_size, bias=bias)

        self.conv3 = nn.Sequential(
            nn.Conv2d(n_feat+2*scale_unetfeats, n_feat+2*scale_unetfeats , 1, padding=0, bias=bias),
            nn.Sigmoid()
        )
        self.channeld3 = conv((n_feat+2*scale_unetfeats)*2, n_feat+2*scale_unetfeats, kernel_size, bias=bias)
    def forward(self, x, mask):
        feat1, feat2, feat3 = x
        mm1, mm2, mm3       = mask
        x1 = feat1+self.body(feat1)*mm1
        x2 = feat2+self.body2(feat2)*mm2
        x3 = feat3+self.body3(feat3)*mm3
        feat1 = self.channeld1(torch.cat([feat1, x1], 1))
        feat2 = self.channeld2(torch.cat([feat2, x2], 1))
        feat3 = self.channeld3(torch.cat([feat3, x3], 1))
        return [feat1, feat2, feat3]

##########################################################################
## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias, scale_unetfeats):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)
        self.up21  = SkipUpSample2(n_feat+scale_unetfeats, 3, 2)
        self.up32  = SkipUpSample2(n_feat+2*scale_unetfeats, 3, 4)
    def forward(self, de0_x, x_img, de1_x, de2_x):
        de2 = self.up32(de2_x)
        de1 = self.up21(de1_x)
        x1  = self.conv1(de0_x)
        de0 = self.conv2(de0_x)
        dm  = de0 + de1 + de2
        img = dm + x_img
        x2  = torch.sigmoid(self.conv3(img))
        x1  = x1*x2
        x1  = x1+de0_x

        return x1, img, dm

##########################################################################
## U-Net

class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff):
        super(Encoder, self).__init__()

        self.encoder_level1 = [CAB(n_feat,                     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level2 = [CAB(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.encoder_level3 = [CAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12  = DownSample(n_feat, scale_unetfeats)
        self.down23  = DownSample(n_feat+scale_unetfeats, scale_unetfeats)

        # Cross Stage Feature Fusion (CSFF)
        if csff:
            self.csff_enc1 = nn.Conv2d(n_feat,                     n_feat,                     kernel_size=1, bias=bias)
            self.csff_enc2 = nn.Conv2d(n_feat+scale_unetfeats,     n_feat+scale_unetfeats,     kernel_size=1, bias=bias)
            self.csff_enc3 = nn.Conv2d(n_feat+(scale_unetfeats*2), n_feat+(scale_unetfeats*2), kernel_size=1, bias=bias)

            self.csff_dec1 = nn.Conv2d(n_feat,                     n_feat,                     kernel_size=1, bias=bias)
            self.csff_dec2 = nn.Conv2d(n_feat+scale_unetfeats,     n_feat+scale_unetfeats,     kernel_size=1, bias=bias)
            self.csff_dec3 = nn.Conv2d(n_feat+(scale_unetfeats*2), n_feat+(scale_unetfeats*2), kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        enc1 = self.encoder_level1(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc1 = enc1 + self.csff_enc1(encoder_outs[0]) + self.csff_dec1(decoder_outs[0])

        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc2 = enc2 + self.csff_enc2(encoder_outs[1]) + self.csff_dec2(decoder_outs[1])

        x = self.down23(enc2)

        enc3 = self.encoder_level3(x)
        if (encoder_outs is not None) and (decoder_outs is not None):
            enc3 = enc3 + self.csff_enc3(encoder_outs[2]) + self.csff_dec3(decoder_outs[2])

        return [enc1, enc2, enc3]

class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False):
        super(Decoder, self).__init__()

        self.decoder_level1 = [CAB(n_feat,                     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level2 = [CAB(n_feat+scale_unetfeats,     kernel_size, reduction, bias=bias, act=act) for _ in range(2)]
        self.decoder_level3 = [CAB(n_feat+(scale_unetfeats*2), kernel_size, reduction, bias=bias, act=act) for _ in range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = CAB(n_feat,                 kernel_size, reduction, bias=bias, act=act)
        self.skip_attn2 = CAB(n_feat+scale_unetfeats, kernel_size, reduction, bias=bias, act=act)

        self.up21  = SkipUpSample(n_feat, scale_unetfeats)
        self.up32  = SkipUpSample(n_feat+scale_unetfeats, scale_unetfeats)

        if csff:
            self.csff_dec1 = nn.Conv2d(n_feat,                     n_feat,                     kernel_size=1, bias=bias)
            self.csff_dec2 = nn.Conv2d(n_feat+scale_unetfeats,     n_feat+scale_unetfeats,     kernel_size=1, bias=bias)
            self.csff_dec3 = nn.Conv2d(n_feat+(scale_unetfeats*2), n_feat+(scale_unetfeats*2), kernel_size=1, bias=bias)

    def forward(self, outs, mm1=None):
        enc1, enc2, enc3 = outs  # mmin0: 32*128*128 1: 48*64*64 2: 64*32*32
        if mm1 == None:
            dec3 = self.decoder_level3(enc3)

            x = self.up32(dec3, self.skip_attn2(enc2))
            dec2 = self.decoder_level2(x)

            x = self.up21(dec2, self.skip_attn1(enc1))
            dec1 = self.decoder_level1(x)
        else:
            dec3 = enc3 + self.csff_dec3(mm1[2])
            dec3 = self.decoder_level3(dec3)

            dec2 = enc2 + self.csff_dec2(mm1[1])
            x = self.up32(dec3, self.skip_attn2(dec2))
            dec2 = self.decoder_level2(x)

            dec1 = enc1 + self.csff_dec1(mm1[0])
            x = self.up21(dec2, self.skip_attn1(dec1))
            dec1 = self.decoder_level1(x)
        return [dec1, dec2, dec3]


##########################################################################
##---------- Resizing Modules ----------
class FuseDown(nn.Module):
    def __init__(self, in_channels, out_channels, s_factor=0.5):
        super(FuseDown, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=s_factor, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x


class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels+s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x


class FuseUp(nn.Module):
    def __init__(self, in_channels, out_channels, s_factor=2):
        super(FuseUp, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=s_factor, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x

class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x, y):
        x = self.up(x)
        x = x + y
        return x

class SkipUpSample2(nn.Module):
    def __init__(self, in_channels, out_chanels, scale_f):
        super(SkipUpSample2, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=scale_f, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, out_chanels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x
##########################################################################
## Original Resolution Block (ORB)
class ORB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(ORB, self).__init__()
        modules_body = []
        modules_body = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

##########################################################################
class ORSNet(nn.Module):
    def __init__(self, n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab):
        super(ORSNet, self).__init__()

        self.orb1 = ORB(n_feat+scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb2 = ORB(n_feat+scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)
        self.orb3 = ORB(n_feat+scale_orsnetfeats, kernel_size, reduction, act, bias, num_cab)

        self.up_enc1 = UpSample(n_feat, scale_unetfeats)
        self.up_dec1 = UpSample(n_feat, scale_unetfeats)

        self.up_enc2 = nn.Sequential(UpSample(n_feat+scale_unetfeats, scale_unetfeats), UpSample(n_feat, scale_unetfeats))
        self.up_dec2 = nn.Sequential(UpSample(n_feat+scale_unetfeats, scale_unetfeats), UpSample(n_feat, scale_unetfeats))

        self.conv_enc1 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc2 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_enc3 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)

        self.conv_dec1 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec2 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)
        self.conv_dec3 = nn.Conv2d(n_feat, n_feat+scale_orsnetfeats, kernel_size=1, bias=bias)

    def forward(self, x, encoder_outs, decoder_outs):
        x = self.orb1(x)
        x = x + self.conv_enc1(encoder_outs[0]) + self.conv_dec1(decoder_outs[0])

        x = self.orb2(x)
        x = x + self.conv_enc2(self.up_enc1(encoder_outs[1])) + self.conv_dec2(self.up_dec1(decoder_outs[1]))

        x = self.orb3(x)
        x = x + self.conv_enc3(self.up_enc2(encoder_outs[2])) + self.conv_dec3(self.up_dec2(decoder_outs[2]))

        return x
##########################################################################
class HRNet(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats):
        super(HRNet, self).__init__()
        self.stage0_channel_down0 = CAB2(2*n_feat, n_feat, kernel_size, reduction, bias=bias, act=act)
        self.stage1_channel_down0 = CAB2(2*n_feat, n_feat, kernel_size, reduction, bias=bias, act=act)
        self.stage1_channel_down1 = CAB2(2*(n_feat+scale_unetfeats), n_feat+scale_unetfeats, kernel_size, reduction, bias=bias, act=act)
        self.stage2_channel_down0 = CAB2(2*n_feat, n_feat, kernel_size, reduction, bias=bias, act=act)
        self.stage2_channel_down1 = CAB2(2*(n_feat+scale_unetfeats), n_feat+scale_unetfeats, kernel_size, reduction, bias=bias, act=act)
        self.stage2_channel_down2 = CAB2(2*(n_feat+2*scale_unetfeats), n_feat+2*scale_unetfeats, kernel_size, reduction, bias=bias, act=act)
        self.stage0_downbranch    = DownSample(n_feat, scale_unetfeats)
        self.stage1_downbranch    = DownSample(n_feat+scale_unetfeats, scale_unetfeats)
        self.stage2_downbranch    = DownSample(n_feat+2*scale_unetfeats, scale_unetfeats)
        self.stage0_downsample    = FuseDown(n_feat, n_feat+scale_unetfeats)
        self.stage0_upsample      = FuseUp(n_feat+scale_unetfeats, n_feat)
        self.stage1_upsample10    = FuseUp(n_feat+scale_unetfeats, n_feat)
        self.stage1_upsample20    = FuseUp(n_feat+2*scale_unetfeats, n_feat, 4)
        self.stage1_upsample21    = FuseUp(n_feat+2*scale_unetfeats, n_feat+scale_unetfeats)
        self.stage1_downsample01  = FuseDown(n_feat, n_feat+scale_unetfeats)
        self.stage1_downsample02  = FuseDown(n_feat, n_feat+2*scale_unetfeats, 0.25)
        self.stage1_downsample12  = FuseDown(n_feat+scale_unetfeats, n_feat+2*scale_unetfeats)
        self.stage2_upsample10    = FuseUp(n_feat+scale_unetfeats, n_feat)
        self.stage2_upsample20    = FuseUp(n_feat+2*scale_unetfeats, n_feat, 4)
        self.stage2_upsample30    = FuseUp(n_feat+3*scale_unetfeats, n_feat, 8)
        self.stage2_upsample21    = FuseUp(n_feat+2*scale_unetfeats, n_feat+scale_unetfeats)
        self.stage2_upsample31    = FuseUp(n_feat+3*scale_unetfeats, n_feat+scale_unetfeats, 4)
        self.stage2_downsample01  = FuseDown(n_feat, n_feat+scale_unetfeats)
        self.stage2_downsample02  = FuseDown(n_feat, n_feat+2*scale_unetfeats, 0.25)
        self.stage2_downsample12  = FuseDown(n_feat+scale_unetfeats, n_feat+2*scale_unetfeats)
        self.stage2_upsample32    = FuseUp(n_feat+3*scale_unetfeats, n_feat+2*scale_unetfeats)
        self.stage2_downsample03  = FuseDown(n_feat, n_feat+3*scale_unetfeats, 0.125)
        self.stage2_downsample13  = FuseDown(n_feat+scale_unetfeats, n_feat+3*scale_unetfeats, 0.25)
        self.stage2_downsample23  = FuseDown(n_feat+2*scale_unetfeats, n_feat+3*scale_unetfeats)

        #self.final_layer         = CAB2(n_feat, 3, kernel_size, reduction, bias=bias, act=act)
        self.final_concat10       = FuseUp(n_feat+scale_unetfeats, n_feat+scale_unetfeats)
        self.final_concat20       = FuseUp(n_feat+2*scale_unetfeats, n_feat+2*scale_unetfeats, 4)
        self.final_concat30       = FuseUp(n_feat+3*scale_unetfeats, n_feat+3*scale_unetfeats, 8)
        self.orb1 = ORB(n_feat, kernel_size, reduction, act, bias, 8)
        self.orb2 = ORB(n_feat, kernel_size, reduction, act, bias, 8)
        self.orb3 = ORB(n_feat, kernel_size, reduction, act, bias, 8)
        self.orb4 = ORB(n_feat, kernel_size, reduction, act, bias, 4)
        self.orb5 = ORB(n_feat, kernel_size, reduction, act, bias, 4)
        self.orb5_1 = ORB(n_feat, kernel_size, reduction, act, bias, 4)
        self.orb6 = ORB(n_feat+scale_unetfeats, kernel_size, reduction, act, bias, 4)
        self.orb6_1 = ORB(n_feat+scale_unetfeats, kernel_size, reduction, act, bias, 4)
        self.orb7 = ORB(n_feat, kernel_size, reduction, act, bias, 4)
        self.orb7_1 = ORB(n_feat, kernel_size, reduction, act, bias, 4)
        self.orb8 = ORB(n_feat+scale_unetfeats, kernel_size, reduction, act, bias, 4)
        self.orb8_1 = ORB(n_feat+scale_unetfeats, kernel_size, reduction, act, bias, 4)
        self.orb9 = ORB(n_feat+2*scale_unetfeats, kernel_size, reduction, act, bias, 4)
        self.orb9_1 = ORB(n_feat+2*scale_unetfeats, kernel_size, reduction, act, bias, 4)
        self.orb10 = ORB(n_feat+2*scale_unetfeats, kernel_size, reduction, act, bias, 4)
        self.orb10_1 = ORB(n_feat+2*scale_unetfeats, kernel_size, reduction, act, bias, 4)
    def forward(self, x, x3_sam, encoder2, decoder2, encoder1, decoder1):
        ##########stage0##########
        res = x
        x3_sam = self.orb4(x3_sam)
        x  = torch.cat([x, x3_sam], 1)
        x  = self.stage0_channel_down0(x)
        x1 = self.stage0_downbranch(x)
        x  = x  + self.stage0_upsample(x1)
        x1 = x1 + self.stage0_downsample(x)
        x  = self.orb1(x)

        ##########stage1##########
        encoder2[0] = self.orb5(encoder2[0])
        encoder1[0] = self.orb5_1(encoder2[0])
        encoder2[1] = self.orb6(encoder2[1])
        encoder1[1] = self.orb6_1(encoder1[1])
        x  = torch.cat([x, encoder2[0]], 1)
        x1 = torch.cat([x1, encoder2[1]], 1)
        x  = self.stage1_channel_down0(x)
        x1 = self.stage1_channel_down1(x1)
        x  = torch.cat([x, encoder1[0]], 1)
        x1 = torch.cat([x1, encoder1[1]], 1)
        x  = self.stage1_channel_down0(x)
        x1 = self.stage1_channel_down1(x1)

        x2 = self.stage1_downbranch(x1)
        x  = x  + self.stage1_upsample10(x1)  + self.stage1_upsample20(x2)
        x1 = x1 + self.stage1_downsample01(x) + self.stage1_upsample21(x2)
        x2 = x2 + self.stage1_downsample02(x) + self.stage1_downsample12(x1)
        x  = self.orb2(x)

        ##########stage2##########
        decoder2[0] = self.orb7(decoder2[0])
        decoder1[0] = self.orb7_1(decoder1[0])
        decoder2[1] = self.orb8(decoder2[1])
        decoder1[1] = self.orb8_1(decoder1[1])
        encoder2[2] = self.orb9(encoder2[2])
        encoder1[2] = self.orb9_1(encoder1[2])
        decoder2[2] = self.orb10(decoder2[2])
        decoder1[2] = self.orb10_1(decoder1[2])
        x  = torch.cat([x, decoder2[0]], 1)
        x1 = torch.cat([x1, decoder2[1]], 1)
        x  = self.stage2_channel_down0(x)
        x1 = self.stage2_channel_down1(x1)
        x  = torch.cat([x, decoder1[0]], 1)
        x1 = torch.cat([x1, decoder1[1]], 1)
        x  = self.stage2_channel_down0(x)
        x1 = self.stage2_channel_down1(x1)
        x2 = torch.cat([x2, encoder2[2]], 1)
        x2 = self.stage2_channel_down2(x2)
        x2 = torch.cat([x2, encoder1[2]], 1)
        x2 = self.stage2_channel_down2(x2)
        x2 = torch.cat([x2, decoder2[2]], 1)
        x2 = self.stage2_channel_down2(x2)
        x2 = torch.cat([x2, decoder1[2]], 1)
        x2 = self.stage2_channel_down2(x2)
        x3 = self.stage2_downbranch(x2)
        x  = x  + self.stage2_upsample10(x1)  + self.stage2_upsample20(x2)   + self.stage2_upsample30(x3) + res
        x1 = x1 + self.stage2_downsample01(x) + self.stage2_upsample21(x2)   + self.stage2_upsample31(x3)
        x2 = x2 + self.stage2_downsample02(x) + self.stage2_downsample12(x1) + self.stage2_upsample32(x3)
        x3 = x3 + self.stage2_downsample03(x) + self.stage2_downsample13(x1)  + self.stage2_downsample23(x2)
        x  = self.orb3(x)
        x1 = self.final_concat10(x1)
        x2 = self.final_concat20(x2)
        x3 = self.final_concat30(x3)
        x  = torch.cat([x, x1], 1)
        x  = torch.cat([x, x2], 1)
        x  = torch.cat([x, x3], 1)
        return x


##########################################################################
class HicarNet(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=32, scale_unetfeats=16, scale_orsnetfeats=16, num_cab=8, kernel_size=3, reduction=4, bias=False):
        super(HicarNet, self).__init__()

        act = nn.PReLU()
        self.shallow_feat1    = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat2    = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))
        self.shallow_feat3    = nn.Sequential(conv(3, n_feat, kernel_size, bias=bias), CAB(n_feat,kernel_size, reduction, bias=bias, act=act))
        self.sum1             =  CAB(3, kernel_size, bias=bias, act=act, reduction=1)
        self.sum2             =  CAB(3, kernel_size, bias=bias, act=act, reduction=1)
        self.sum3             =  CAB(3, kernel_size, bias=bias, act=act, reduction=1)
        self.patch_attention  =  PAB(n_feat, kernel_size, reduction, scale_unetfeats, bias=bias, act=act)
        self.patch_attention2 =  PAB(n_feat, kernel_size, reduction, scale_unetfeats, bias=bias, act=act)
        # Cross Stage Feature Fusion (CSFF)
        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        self.stage1_encoder_mm = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=False)
        self.stage1_decoder_mm = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        self.stage1_decoder_mmin = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        self.covmm =  nn.Conv2d(n_feat, 3, kernel_size=1)
        self.stage2_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)
        self.stage2_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        self.stage2_decoder_mm = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, csff=True)

        self.stage3_orsnet = ORSNet(n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab)
        self.stage3_hrnet  = HRNet(n_feat, kernel_size, reduction, act, bias, scale_unetfeats)
        #self.mask_orsnet   = ORSNet(n_feat, scale_orsnetfeats, kernel_size, reduction, act, bias, scale_unetfeats, num_cab)

        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias, scale_unetfeats=scale_unetfeats)
        self.sam23 = SAM(n_feat, kernel_size=1, bias=bias, scale_unetfeats=scale_unetfeats)

        self.concat12  = conv(n_feat*2, n_feat, kernel_size, bias=bias)
        self.concat23  = conv(n_feat*2, n_feat+scale_orsnetfeats, kernel_size, bias=bias)
        self.concat3m  = conv(2*n_feat+scale_orsnetfeats, 2*n_feat+scale_orsnetfeats, kernel_size, bias=bias)
        #self.mask23   = conv(n_feat*2, n_feat+scale_orsnetfeats, kernel_size, bias=bias)
        self.tail      = conv(4*n_feat+6*scale_unetfeats, 3, kernel_size, bias=bias)
        #self.mask_tail = conv(n_feat+scale_orsnetfeats, 3, kernel_size, bias=bias)

    def forward(self, x3_img, flag=False):
        # Original-resolution Image for Stage 3
        H = x3_img.size(2)
        W = x3_img.size(3)
        # Multi-Patch Hierarchy: Split Image into four non-overlapping patches
        # Two Patches for Stage 2
        x2top_img  = x3_img[:,:,0:int(H/2),:]
        x2bot_img  = x3_img[:,:,int(H/2):H,:]

        # Four Patches for Stage 1
        x1ltop_img = x2top_img[:,:,:,0:int(W/2)]  # 3*128*128
        x1rtop_img = x2top_img[:,:,:,int(W/2):W]
        x1lbot_img = x2bot_img[:,:,:,0:int(W/2)]
        x1rbot_img = x2bot_img[:,:,:,int(W/2):W]

        ##-------------------------------------------
        ##-------------- Stage 1---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x1ltop = self.shallow_feat1(x1ltop_img)  # 32*128*128
        x1rtop = self.shallow_feat1(x1rtop_img)  # 32*128*128
        x1lbot = self.shallow_feat1(x1lbot_img)  # 32*128*128
        x1rbot = self.shallow_feat1(x1rbot_img)  # 32*128*128

        ## Process features of all 4 patches with Encoder of Stage 1
        feat1_ltop = self.stage1_encoder(x1ltop)  # 0: 32*128*128 1: 48*64*64 2: 64*32*32
        feat1_rtop = self.stage1_encoder(x1rtop)
        feat1_lbot = self.stage1_encoder(x1lbot)
        feat1_rbot = self.stage1_encoder(x1rbot)

        ## Concat deep features
        feat1_top = [torch.cat((k, v), 3) for k, v in zip(feat1_ltop, feat1_rtop)]  # 0: 32*128*256 1: 48*64*128 2: 64*32*64
        feat1_bot = [torch.cat((k, v), 3) for k, v in zip(feat1_lbot, feat1_rbot)]
        feat1     = [torch.cat((k, v), 2) for k, v in zip(feat1_top, feat1_bot)]

        #mm1top   = self.stage1_decoder_mm(feat1_top) # 0: 32*128*256 1: 48*64*128 2: 64*32*64
        #mm1bot   = self.stage1_decoder_mm(feat1_bot)
        mm1ltop  = self.stage1_decoder_mmin(feat1_ltop) # 0: 32*128*128 1: 48*64*64 2: 64*32*32
        mm1rtop  = self.stage1_decoder_mmin(feat1_rtop)
        mm1lbot  = self.stage1_decoder_mmin(feat1_lbot)
        mm1rbot  = self.stage1_decoder_mmin(feat1_rbot)
        mm1top   = [torch.cat((k, v), 3) for k, v in zip(mm1ltop, mm1rtop)]
        mm1bot  = [torch.cat((k, v), 3) for k, v in zip(mm1lbot, mm1rbot)]
        mm1      = [torch.cat((k, v), 2) for k, v in zip(mm1top, mm1bot)]

        ## Pass features through Decoder of Stage 1
        res1_top = self.stage1_decoder(feat1_top)  # 0: 32*128*256 1: 48*64*128 2: 64*32*64
        res1_bot = self.stage1_decoder(feat1_bot)

        ## Add Patch Attention Module
        if flag == True:
            feat1_top = self.patch_attention(res1_top, mm1top)
            feat1_bot = self.patch_attention(res1_bot, mm1bot)

        res1     = [torch.cat((k, v), 2) for k, v in zip(res1_top, res1_bot)]


        ## Apply Supervised Attention Module (SAM)

        x2top_samfeats, stage1_img_top, stag1_res_top = self.sam12(res1_top[0], x2top_img, res1_top[1], res1_top[2])  # 40*128*256, 3*128*256, 3*128*256
        x2bot_samfeats, stage1_img_bot, stag1_res_bot = self.sam12(res1_bot[0], x2bot_img, res1_bot[1], res1_bot[2])

        ## Output image at Stage 1
        stage1_img = torch.cat([stage1_img_top, stage1_img_bot], 2)
        stage1_res = torch.cat((stag1_res_top, stag1_res_bot), 2)
        ##-------------------------------------------
        ##-------------- Stage 2---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x2top  = self.shallow_feat2(x2top_img)
        x2bot  = self.shallow_feat2(x2bot_img)

        ## Concatenate SAM features of Stage 1 with shallow features of Stage 2
        x2top_cat = self.concat12(torch.cat([x2top, x2top_samfeats], 1))
        x2bot_cat = self.concat12(torch.cat([x2bot, x2bot_samfeats], 1))

        ## Process features of both patches with Encoder of Stage 2
        feat2_top = self.stage2_encoder(x2top_cat, feat1_top, res1_top)
        feat2_bot = self.stage2_encoder(x2bot_cat, feat1_bot, res1_bot)

        ## Concat deep features
        feat2 = [torch.cat((k, v), 2) for k, v in zip(feat2_top, feat2_bot)]
        mm2top  = self.stage2_decoder_mm(feat2_top, mm1top)
        mm2bot  = self.stage2_decoder_mm(feat2_bot, mm1bot)
        mm2      = [torch.cat((k, v), 2) for k, v in zip(mm2top, mm2bot)]
        mm3     = self.covmm(mm2[0])
        if flag == True:
            feat2 = self.patch_attention2(feat2, mm2)

        ## Pass features through Decoder of Stage 2
        res2 = self.stage2_decoder(feat2)

        ## Apply SAM
        x3_samfeats, stage2_img, stage2_res = self.sam23(res2[0], x3_img, res2[1], res2[2])

        ##-------------------------------------------
        ##-------------- Stage 3---------------------
        ##-------------------------------------------
        ## Compute Shallow Features
        x3     = self.shallow_feat3(x3_img)


        ## Concatenate SAM features of Stage 2 with shallow features of Stage 3
        #x3_cat = self.concat23(torch.cat([x3, x3_samfeats], 1))  # 48*256*256


        stage3_img = self.stage3_hrnet(x3, x3_samfeats, feat2, res2, feat1, res1)  # feat2, res2: 0:32*256*256 1:48*128*128 2:64*64*64

        stage3_img = self.tail(stage3_img)
        out_img    = self.sum1(stage1_img)+self.sum2(stage2_img)+self.sum3(stage3_img)
        #out_img    = 0.15*stage1_res+0.3*stage2_res+0.6*stage3_img+x3_img
        return [out_img, stage2_img, stage1_img, mm3]
