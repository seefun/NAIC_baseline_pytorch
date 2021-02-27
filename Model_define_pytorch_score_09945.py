#!/usr/bin/env python3
"""An Implement of an autoencoder with pytorch.
This is the template code for 2020 NIAC https://naic.pcl.ac.cn/.
The code is based on the sample code with tensorflow for 2020 NIAC and it can only run with GPUS.
If you have any questions, please contact me with https://github.com/xufana7/AutoEncoder-with-pytorch
Author, Fan xu Aug 2020
"""
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import OrderedDict

# from torchvision.models import resnet18

channel_num = 768

# This part implement the quantization and dequantization operations.
# The output of the encoder must be the bitstream.
def Num2Bit(Num, B):
    Num_ = Num.type(torch.uint8)

    def integer2bit(integer, num_bits=B * 2):
        dtype = integer.type()
        exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
        exponent_bits = exponent_bits.repeat(integer.shape + (1,))
        out = integer.unsqueeze(-1) // 2 ** exponent_bits
        return (out - (out % 1)) % 2

    bit = integer2bit(Num_)
    bit = (bit[:, :, B:]).reshape(-1, Num_.shape[1] * B)
    return bit.type(torch.float32)


def Bit2Num(Bit, B):
    Bit_ = Bit.type(torch.float32)
    Bit_ = torch.reshape(Bit_, [-1, int(Bit_.shape[1] / B), B])
    num = torch.zeros(Bit_[:, :, 1].shape).cuda()
    for i in range(B):
        num = num + Bit_[:, :, i] * 2 ** (B - 1 - i)
    return num


class Quantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = torch.round(x * step - 0.5)
        out = Num2Bit(out, B)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of constant arguments to forward must be None.
        # Gradient of a number is the sum of its four bits.
        b, _ = grad_output.shape
        grad_num = torch.sum(grad_output.reshape(b, -1, ctx.constant), dim=2)
        return grad_num, None


class Dequantization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, B):
        ctx.constant = B
        step = 2 ** B
        out = Bit2Num(x, B)
        out = (out + 0.5) / step
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        # repeat the gradient of a Num for four time.
        #b, c = grad_output.shape
        #grad_bit = grad_output.repeat(1, 1, ctx.constant) 
        #return torch.reshape(grad_bit, (-1, c * ctx.constant)), None
        grad_bit = grad_output.repeat_interleave(ctx.constant, dim=1)
        return grad_bit, None


class QuantizationLayer(nn.Module):

    def __init__(self, B):
        super(QuantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Quantization.apply(x, self.B)
        return out


class DequantizationLayer(nn.Module):

    def __init__(self, B):
        super(DequantizationLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = Dequantization.apply(x, self.B)
        return out
    
    
class FakeQuantOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_bits=4):
        step = 2 ** num_bits
        out = torch.round(x * step - 0.5)
        x = (out + 0.5) / step
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # straight through estimator
        return grad_output, None

    
class FakeQuantLayer(nn.Module):

    def __init__(self, B):
        super(FakeQuantLayer, self).__init__()
        self.B = B

    def forward(self, x):
        out = FakeQuantOp.apply(x, self.B)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)

class SEBlock(nn.Module):
    def __init__(self, in_ch, r=8):
        super(SEBlock, self).__init__()

        self.linear_1 = nn.Linear(in_ch, in_ch//r)
        self.linear_2 = nn.Linear(in_ch//r, in_ch)

    def forward(self, x):
        input_x = x
        x = F.relu(self.linear_1(x), inplace=True)
        x = self.linear_2(x)
        x = torch.sigmoid(x)
        x = input_x * x
        return x

    
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x *( torch.tanh(F.softplus(x)))


class Swish(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)
        
        
class ActBNConv(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ActBNConv, self).__init__(OrderedDict([
            ('act', Swish()),
            ('bn', nn.BatchNorm2d(in_planes)),
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False))
        ]))


class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


        
class BasicBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = ActBNConv(in_channels,in_channels,3)
        self.conv2 = ActBNConv(in_channels,in_channels,3)
        self.attention = SCSEModule(in_channels)
        self.identity = nn.Identity()

    def forward(self, x):
        identity = self.identity(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention(x)
        x = x + identity
        return x
    
class BasicBlock_2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ActBNConv(in_channels,out_channels,3)
        self.conv2 = ActBNConv(out_channels,out_channels,3)
        self.attention = SCSEModule(out_channels)
        self.identity = nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention(x)
        return x


# class Encoder(nn.Module):
#     B = 4

#     def __init__(self, feedback_bits, quantization=True):
#         super(Encoder, self).__init__()
        
#         self.model = resnet18(pretrained=True)
#         weight_rgb = self.model.conv1.weight
#         weight_grey = weight_rgb[:,:2,:,:] + weight_rgb[:,-1,:,:].unsqueeze(1) / 2.0
#         self.model.conv1 = nn.Conv2d(2, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
#         self.model.conv1.weight = torch.nn.Parameter(weight_grey)
#         self.model.fc = nn.Linear(512, 32)
        
#         self.sig = nn.Sigmoid()
#         self.quantize = QuantizationLayer(self.B)
#         self.quantization = quantization 
#         self.fake_quantize = FakeQuantLayer(self.B)

#     def forward(self, x):
#         out = self.model(x)
#         out = self.sig(out)
#         if self.quantization:
#             out = self.quantize(out)
#         else:
#             out = self.fake_quantize(out)
#         return out


class Encoder(nn.Module):
    B = 4

    def __init__(self, feedback_bits, quantization=True):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(2, channel_num, kernel_size=3, stride=1, padding=1, bias=True)
        self.encoder1 = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ActBNConv(channel_num, channel_num, 3)),
            ("conv1x9_bn", ActBNConv(channel_num, channel_num, [1, 9])),
            ("conv9x1_bn", ActBNConv(channel_num, channel_num, [9, 1])),
        ]))
        self.encoder2 = ActBNConv(channel_num, channel_num//4*3, 3)
        self.encoder3 = ActBNConv(channel_num, channel_num//4, 5)
        self.encoder_conv = nn.Sequential(OrderedDict([
            ("conv1x1_bn_1", ActBNConv(channel_num*3, channel_num, 1)),
            ("EncoderBlock", BasicBlock_2(channel_num, channel_num)),
            ("conv1x1_bn_2", ActBNConv(channel_num, 2, 1)),
        ]))
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.fc = nn.Linear(1024, int(feedback_bits / self.B))
        self.sig = nn.Sigmoid()
        self.quantize = QuantizationLayer(self.B)
        self.quantization = quantization 
        self.fake_quantize = FakeQuantLayer(self.B)

    def forward(self, x):
        x = self.conv1(x)
        encode1 = self.encoder1(x)
        encode2 = self.encoder2(x)
        encode3 = self.encoder3(x)
        out = torch.cat([encode1, encode2, encode3, x], dim=1)
        out = self.encoder_conv(out)
        out = self.relu(out)
        out = out.view(-1, 1024)
        out = self.fc(out)
        out = self.sig(out)
        if self.quantization == 'check':
            out = out
        elif self.quantization:
            out = self.quantize(out)
        else:
            out = self.fake_quantize(out)
        return out

    
class Decoder(nn.Module):
    B = 4

    def __init__(self, feedback_bits, quantization=True):
        super(Decoder, self).__init__()
        self.feedback_bits = feedback_bits
        self.dequantize = DequantizationLayer(self.B)
        self.offset = nn.Sequential(
            nn.Linear(int(feedback_bits / self.B), 128), nn.LeakyReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, int(feedback_bits / self.B), nn.Sigmoid()),
            )
        self.fc = nn.Linear(int(feedback_bits / self.B), 1024)
        self.se = SEBlock(1024)
        self.bn = nn.BatchNorm1d(1024)
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        decoder = OrderedDict([
            ("conv5x5", nn.Conv2d(2, channel_num, kernel_size=5, stride=1, padding=2, bias=True)),
            ("DecoderBlock1", BasicBlock(channel_num)),
            ("DecoderBlock2", BasicBlock(channel_num)),
            ("DecoderBlock3", BasicBlock(channel_num)),
            ("DecoderBlock4", BasicBlock(channel_num)),
            ("DecoderBlock5", BasicBlock_2(channel_num, channel_num//4)),
            ("DecoderBlock6", BasicBlock(channel_num//4)),
            ("DecoderBlock7", BasicBlock(channel_num//4)),
            ("DecoderBlock8", BasicBlock(channel_num//4)),
            ("DecoderBlock9", BasicBlock_2(channel_num//4, channel_num//8)),
            ("DecoderBlock10", BasicBlock(channel_num//8)),
            ("DecoderBlock11", BasicBlock(channel_num//8)),
        ])
        self.decoder_feature = nn.Sequential(decoder)
        self.out_cov = ActBNConv(channel_num//8, 2, 3)
        self.sig = nn.Sigmoid()
        self.quantization = quantization        

    def forward(self, x):
        if self.quantization == 'check':
            out = x
        elif self.quantization:
            out = self.dequantize(x)
        else:
            out = x
        out = out.view(-1, int(self.feedback_bits / self.B))
        if self.quantization:
            out = out + self.offset(out)/(2**(self.B))
        out = self.fc(out)
        out = self.relu(out)
        out = self.se(out)
        out = self.bn(out)
        out = out.view(-1, 2, 16, 32)
        out = self.decoder_feature(out)
        out = self.out_cov(out)
        out = self.sig(out)
        return out


# Note: Do not modify following class and keep it in your submission.
# feedback_bits is 128 by default.
class AutoEncoder(nn.Module):

    def __init__(self, feedback_bits):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(feedback_bits)
        self.decoder = Decoder(feedback_bits)

    def forward(self, x):
        feature = self.encoder(x)
        out = self.decoder(feature)
        return out


def NMSE(x, x_hat):
    x_real = np.reshape(x[:, :, :, 0], (len(x), -1))
    x_imag = np.reshape(x[:, :, :, 1], (len(x), -1))
    x_hat_real = np.reshape(x_hat[:, :, :, 0], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, :, :, 1], (len(x_hat), -1))
    x_C = x_real - 0.5 + 1j * (x_imag - 0.5)
    x_hat_C = x_hat_real - 0.5 + 1j * (x_hat_imag - 0.5)
    power = np.sum(abs(x_C) ** 2, axis=1)
    mse = np.sum(abs(x_C - x_hat_C) ** 2, axis=1)
    nmse = np.mean(mse / power)
    return nmse

def NMSE_cuda(x, x_hat):
    x_real = x[:, 0, :, :].view(len(x),-1) - 0.5
    x_imag = x[:, 1, :, :].view(len(x),-1) - 0.5
    x_hat_real = x_hat[:, 0, :, :].view(len(x_hat), -1) - 0.5
    x_hat_imag = x_hat[:, 1, :, :].view(len(x_hat), -1) - 0.5
    power = torch.sum(x_real**2 + x_imag**2, axis=1)
    mse = torch.sum((x_real-x_hat_real)**2 + (x_imag-x_hat_imag)**2, axis=1)
    nmse = mse/power
    return nmse
    

class NMSELoss(nn.Module):
    def __init__(self, reduction='sum'):
        super(NMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, x_hat, x):
        nmse = NMSE_cuda(x, x_hat)
        if self.reduction == 'mean':
            nmse = torch.mean(nmse) 
        else:
            nmse = torch.sum(nmse)
        return nmse
    
    
class MSE_NMSELoss(nn.Module):
    def __init__(self, alpha = 100., reduction='sum'):
        super(MSE_NMSELoss, self).__init__()
        self.reduction = reduction
        self.MSELoss = nn.MSELoss(reduction=self.reduction) 
        self.alpha = alpha

    def forward(self, x_hat, x):
        nmse = NMSE_cuda(x, x_hat)
        if self.reduction == 'mean':
            nmse = torch.mean(nmse) 
        else:
            nmse = torch.sum(nmse)
        loss = nmse + self.alpha * self.MSELoss(x, x_hat)
        return loss
       

def Score(NMSE):
    score = 1 - NMSE
    return score


# dataLoader
class DatasetFolder(Dataset):

    def __init__(self, matData):
        self.matdata = matData

    def __len__(self):
        return self.matdata.shape[0]
    
    def __getitem__(self, index):
        return self.matdata[index] #, self.matdata[index]
