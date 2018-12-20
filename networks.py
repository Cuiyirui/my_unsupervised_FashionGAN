"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import numpy as np
from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from torch.nn import init
import functools
import torchvision.models as tvmodels

try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass

##################################################################################
# Discriminator
##################################################################################

class MsImageDis(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, input_dim, params):
        super(MsImageDis, self).__init__()
        self.n_layer = params['n_layer']
        self.gan_type = params['gan_type']
        self.dim = params['dim']
        self.norm = params['norm']
        self.activ = params['activ']
        self.num_scales = params['num_scales']
        self.pad_type = params['pad_type']
        self.input_dim = input_dim
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1)**2) # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

##################################################################################
# Generator
##################################################################################

class AdaINGen(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, params):
        super(AdaINGen, self).__init__()
        dim = params['dim']
        style_dim = params['style_dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']
        mlp_dim = params['mlp_dim']

        # style encoder
        self.enc_style = StyleEncoder(4, input_dim, dim, style_dim, norm='none', activ=activ, pad_type=pad_type)

        # content encoder
        self.enc_content = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.dec = Decoder(n_downsample, n_res, self.enc_content.output_dim, input_dim, res_norm='adain', activ=activ, pad_type=pad_type)

        # MLP to generate AdaIN parameters
        self.mlp = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)

    def forward(self, images):
        # reconstruct an image
        content, style_fake = self.encode(images)
        images_recon = self.decode(content, style_fake)
        return images_recon

    def encode(self, images):
        # encode an image to its content and style codes
        style_fake = self.enc_style(images)
        content = self.enc_content(images)
        return content, style_fake

    def decode(self, content, style):
        # decode content and style codes to an image
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params


class VAEGen(nn.Module):
    # VAE architecture
    def __init__(self, input_dim, params):
        super(VAEGen, self).__init__()
        dim = params['dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']

        # content encoder
        self.enc = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.dec = Decoder(n_downsample, n_res, self.enc.output_dim, input_dim, res_norm='in', activ=activ, pad_type=pad_type)

    def forward(self, images):
        # This is a reduced VAE implementation where we assume the outputs are multivariate Gaussian distribution with mean = hiddens and std_dev = all ones.
        hiddens = self.encode(images)
        if self.training == True:
            noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
            images_recon = self.decode(hiddens + noise)
        else:
            images_recon = self.decode(hiddens)
        return images_recon, hiddens

    def encode(self, images):
        hiddens = self.enc(images)
        noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
        return hiddens, noise

    def decode(self, hiddens):
        images = self.dec(hiddens)
        return images


##################################################################################
# Encoder and Decoders
##################################################################################

class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(StyleEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)] # global average pooling
        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero'):
        super(Decoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

##################################################################################
# Sequential Models
##################################################################################
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out

##################################################################################
# VGG network definition
##################################################################################
class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        # relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        # relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        # relu3_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        h = F.relu(self.conv4_2(h), inplace=True)
        h = F.relu(self.conv4_3(h), inplace=True)
        # relu4_3 = h

        h = F.relu(self.conv5_1(h), inplace=True)
        h = F.relu(self.conv5_2(h), inplace=True)
        h = F.relu(self.conv5_3(h), inplace=True)
        relu5_3 = h

        return relu5_3
        # return [relu1_2, relu2_2, relu3_3, relu4_3]

##################################################################################
# Normalization layers
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

# my components
def upsampleLayer(inplanes, outplanes, upsample='basic', padding_type='zero'):
    # padding_type = 'zero'
    if upsample == 'basic':
        upconv = [nn.ConvTranspose2d(
            inplanes, outplanes, kernel_size=4, stride=2, padding=1)]
    elif upsample == 'bilinear':
        upconv = [nn.Upsample(scale_factor=2, mode='bilinear'),
                  nn.ReflectionPad2d(1),
                  nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0)]
    else:
        raise NotImplementedError(
            'upsample layer [%s] not implemented' % upsample)
    return upconv



# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetBlock(nn.Module):
    def __init__(self, input_nc, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic', padding_type='zero'):
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        p = 0
        downconv = []
        if padding_type == 'reflect':
            downconv += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            downconv += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)
        downconv += [nn.Conv2d(input_nc, inner_nc,
                               kernel_size=4, stride=2, padding=p)]
        # downsample is different from upsample
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc) if norm_layer is not None else None
        uprelu = nl_layer()
        upnorm = norm_layer(outer_nc) if norm_layer is not None else None

        if outermost:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = downconv
            up = [uprelu] + upconv + [nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = upsampleLayer(
                inner_nc, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            up = [uprelu] + upconv
            if upnorm is not None:
                up += [upnorm]
            model = down + up
        else:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            if downnorm is not None:
                down += [downnorm]
            up = [uprelu] + upconv
            if upnorm is not None:
                up += [upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)

class UnetBlock_with_z(nn.Module):
    def __init__(self, input_nc, outer_nc, inner_nc, nz=0,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic', padding_type='zero'):
        super(UnetBlock_with_z, self).__init__()
        p = 0
        downconv = []
        if padding_type == 'reflect':
            downconv += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            downconv += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)

        self.outermost = outermost
        self.innermost = innermost
        self.nz = nz
        input_nc = input_nc + nz
        downconv += [nn.Conv2d(input_nc, inner_nc,
                               kernel_size=4, stride=2, padding=p)]
        # downsample is different from upsample
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu = nl_layer()

        if outermost:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = downconv
            up = [uprelu] + upconv + [nn.Tanh()]
        elif innermost:
            upconv = upsampleLayer(
                inner_nc, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            up = [uprelu] + upconv
            if norm_layer is not None:
                up += [norm_layer(outer_nc)]
        else:
            upconv = upsampleLayer(
                inner_nc * 2, outer_nc, upsample=upsample, padding_type=padding_type)
            down = [downrelu] + downconv
            if norm_layer is not None:
                down += [norm_layer(inner_nc)]
            up = [uprelu] + upconv

            if norm_layer is not None:
                up += [norm_layer(outer_nc)]

            if use_dropout:
                up += [nn.Dropout(0.5)]
        self.down = nn.Sequential(*down)
        self.submodule = submodule
        self.up = nn.Sequential(*up)

    def forward(self, x, z):
        # print(x.size())
        if self.nz > 0:
            z_img = z.view(z.size(0), z.size(1), 1, 1).expand(
                z.size(0), z.size(1), x.size(2), x.size(3))
            x_and_z = torch.cat([x, z_img], 1)
        else:
            x_and_z = x

        if self.outermost:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return self.up(x2)
        elif self.innermost:
            x1 = self.up(self.down(x_and_z))
            return torch.cat([x1, x], 1)
        else:
            x1 = self.down(x_and_z)
            x2 = self.submodule(x1, z)
            return torch.cat([self.up(x2), x], 1)


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class G_Unet_add_input(nn.Module):
    def __init__(self, input_nc, output_nc, nz, num_downs, ngf=64,
                 norm_layer=None, nl_layer=None, use_dropout=False,
                 gpu_ids=[], upsample='basic'):
        super(G_Unet_add_input, self).__init__()
        self.gpu_ids = gpu_ids
        self.nz = nz
        # currently support only input_nc == output_nc
        # assert(input_nc == output_nc)
        max_nchn = 8
        # construct unet structure
        unet_block = UnetBlock(ngf * max_nchn, ngf * max_nchn, ngf * max_nchn,
                               innermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        for i in range(num_downs - 5):
            unet_block = UnetBlock(ngf * max_nchn, ngf * max_nchn, ngf * max_nchn, unet_block,
                                   norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout, upsample=upsample)
        unet_block = UnetBlock(ngf * 4, ngf * 4, ngf * max_nchn, unet_block,
                               norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock(ngf * 2, ngf * 2, ngf * 4, unet_block,
                               norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock(ngf, ngf, ngf * 2, unet_block,
                               norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock(input_nc + nz, output_nc, ngf, unet_block,
                               outermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)

        self.model = unet_block

    def forward(self, x, z=None):
        if self.nz > 0:
            z_img = z.view(z.size(0), z.size(1), 1, 1).expand(
                z.size(0), z.size(1), x.size(2), x.size(3))
            x_with_z = torch.cat([x, z_img], 1)
        else:
            x_with_z = x  # no z

        return self.model(x_with_z)


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class G_Unet_add_all(nn.Module):
    def __init__(self, input_nc, output_nc, nz, num_downs, ngf=64,
                 norm_layer=None, nl_layer=None, use_dropout=False, gpu_ids=[], upsample='basic'):
        super(G_Unet_add_all, self).__init__()
        self.gpu_ids = gpu_ids
        self.nz = nz
        # construct unet structure
        unet_block = UnetBlock_with_z(ngf * 8, ngf * 8, ngf * 8, nz, None, innermost=True,
                                      norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(ngf * 8, ngf * 8, ngf * 8, nz, unet_block,
                                      norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout, upsample=upsample)
        for i in range(num_downs - 6):
            unet_block = UnetBlock_with_z(ngf * 8, ngf * 8, ngf * 8, nz, unet_block,
                                          norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout, upsample=upsample)
        unet_block = UnetBlock_with_z(ngf * 4, ngf * 4, ngf * 8, nz, unet_block,
                                      norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(ngf * 2, ngf * 2, ngf * 4, nz, unet_block,
                                      norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(
            ngf, ngf, ngf * 2, nz, unet_block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock_with_z(input_nc, output_nc, ngf, nz, unet_block,
                                      outermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        self.model = unet_block

    def forward(self, x, z):
        return self.model(x, z)

# define generator
def define_G(input_nc, output_nc, nz, ngf,
             which_model_netG='unet_256', norm='batch', nl='relu',
             use_dropout=False, init_type='xavier', gpu_ids=[], where_add='all', upsample='bilinear'):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(layer_type=norm)
    nl_layer = get_non_linearity(layer_type=nl)
    # upsample = 'bilinear'
    if use_gpu:
        assert(torch.cuda.is_available())

    if nz == 0:
        where_add = 'input'

    if which_model_netG == 'unet_64' and where_add == 'input':
        netG = G_Unet_add_input(input_nc, output_nc, nz, 6, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                                use_dropout=use_dropout, gpu_ids=gpu_ids, upsample=upsample)
    elif which_model_netG == 'unet_128' and where_add == 'input':
        netG = G_Unet_add_input(input_nc, output_nc, nz, 7, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                                use_dropout=use_dropout, gpu_ids=gpu_ids, upsample=upsample)
    elif which_model_netG == 'unet_256' and where_add == 'input':
        netG = G_Unet_add_input(input_nc, output_nc, nz, 8, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                                use_dropout=use_dropout, gpu_ids=gpu_ids, upsample=upsample)
    elif which_model_netG == 'unet_512' and where_add == 'input':
        netG = G_Unet_add_input(input_nc, output_nc, nz, 9, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                                use_dropout=use_dropout, gpu_ids=gpu_ids, upsample=upsample)
    elif which_model_netG == 'unet_1024' and where_add == 'input':
        netG = G_Unet_add_input(input_nc, output_nc, nz, 10, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                                use_dropout=use_dropout, gpu_ids=gpu_ids, upsample=upsample)
    elif which_model_netG == 'unet_64' and where_add == 'all':
        netG = G_Unet_add_input(input_nc, output_nc, nz, 6, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                                use_dropout=use_dropout, gpu_ids=gpu_ids, upsample=upsample)
    elif which_model_netG == 'unet_128' and where_add == 'all':
        netG = G_Unet_add_all(input_nc, output_nc, nz, 7, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                              use_dropout=use_dropout, gpu_ids=gpu_ids, upsample=upsample)
    elif which_model_netG == 'unet_256' and where_add == 'all':
        netG = G_Unet_add_all(input_nc, output_nc, nz, 8, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                              use_dropout=use_dropout, gpu_ids=gpu_ids, upsample=upsample)
    elif which_model_netG == 'unet_512' and where_add == 'all':
        netG = G_Unet_add_input(input_nc, output_nc, nz, 9, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                                use_dropout=use_dropout, gpu_ids=gpu_ids, upsample=upsample)
    elif which_model_netG == 'unet_1024' and where_add == 'all':
        netG = G_Unet_add_input(input_nc, output_nc, nz, 10, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                                use_dropout=use_dropout, gpu_ids=gpu_ids, upsample=upsample)
    else:
        raise NotImplementedError(
            'Generator model name [%s] is not recognized' % which_model_netG)

    if len(gpu_ids) > 0:
        print (gpu_ids)
        #input()
        netG.cuda(gpu_ids[0])

    init_weights(netG, init_type=init_type)
    return netG


class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlock, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [conv3x3(inplanes, inplanes)]
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [convMeanpool(inplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = meanpoolConv(inplanes, outplanes)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out

class E_ResNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, ndf=64, n_blocks=4,
                 norm_layer=None, nl_layer=None, gpu_ids=[], vaeLike=False):
        super(E_ResNet, self).__init__()
        self.gpu_ids = gpu_ids
        self.vaeLike = vaeLike
        max_ndf = 4
        conv_layers = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=True)]
        for n in range(1, n_blocks):
            input_ndf = ndf * min(max_ndf, n)
            output_ndf = ndf * min(max_ndf, n + 1)
            conv_layers += [BasicBlock(input_ndf,
                                       output_ndf, norm_layer, nl_layer)]
        conv_layers += [nl_layer(), nn.AvgPool2d(8)]
        if vaeLike:
            self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
            self.fcVar = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc(conv_flat)
        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        else:
            return output
        return output


class E_NLayers(nn.Module):
    def __init__(self, input_nc, output_nc=1, ndf=64, n_layers=3,
                 norm_layer=None, nl_layer=None, gpu_ids=[], vaeLike=False):
        super(E_NLayers, self).__init__()
        self.gpu_ids = gpu_ids
        self.vaeLike = vaeLike

        kw, padw = 4, 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw,
                              stride=2, padding=padw), nl_layer()]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 4)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw)]
            if norm_layer is not None:
                sequence += [norm_layer(ndf * nf_mult)]
            sequence += [nl_layer()]
        sequence += [nn.AvgPool2d(8)]
        self.conv = nn.Sequential(*sequence)
        self.fc = nn.Sequential(*[nn.Linear(ndf * nf_mult, output_nc)])
        if vaeLike:
            self.fcVar = nn.Sequential(*[nn.Linear(ndf * nf_mult, output_nc)])

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc(conv_flat)
        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        return output


# define encoder
def define_E(input_nc, output_nc, ndf, which_model_netE='resnet_256',
             norm='batch', nl='lrelu',
             init_type='xavier', gpu_ids=[], vaeLike=False):
    netE = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(layer_type=norm)
    nl = 'lrelu'  # use leaky relu for E
    nl_layer = get_non_linearity(layer_type=nl)
    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netE == 'resnet_32':
        netE = E_ResNet(input_nc, output_nc, ndf, n_blocks=2, norm_layer=norm_layer,
                        nl_layer=nl_layer, gpu_ids=gpu_ids, vaeLike=vaeLike)
    elif which_model_netE == 'resnet_64':
        netE = E_ResNet(input_nc, output_nc, ndf, n_blocks=3, norm_layer=norm_layer,
                        nl_layer=nl_layer, gpu_ids=gpu_ids, vaeLike=vaeLike)
    elif which_model_netE == 'resnet_128':
        netE = E_ResNet(input_nc, output_nc, ndf, n_blocks=4, norm_layer=norm_layer,
                        nl_layer=nl_layer, gpu_ids=gpu_ids, vaeLike=vaeLike)
    elif which_model_netE == 'resnet_256':
        netE = E_ResNet(input_nc, output_nc, ndf, n_blocks=5, norm_layer=norm_layer,
                        nl_layer=nl_layer, gpu_ids=gpu_ids, vaeLike=vaeLike)
    elif which_model_netE == 'resnet_512':
        netE = E_ResNet(input_nc, output_nc, ndf, n_blocks=6, norm_layer=norm_layer,
                        nl_layer=nl_layer, gpu_ids=gpu_ids, vaeLike=vaeLike)
    elif which_model_netE == 'resnet_1024':
        netE = E_ResNet(input_nc, output_nc, ndf, n_blocks=7, norm_layer=norm_layer,
                        nl_layer=nl_layer, gpu_ids=gpu_ids, vaeLike=vaeLike)

    elif which_model_netE == 'conv_128':
        netE = E_NLayers(input_nc, output_nc, ndf, n_layers=4, norm_layer=norm_layer,
                         nl_layer=nl_layer, gpu_ids=gpu_ids, vaeLike=vaeLike)
    elif which_model_netE == 'conv_256':
        netE = E_NLayers(input_nc, output_nc, ndf, n_layers=5, norm_layer=norm_layer,
                         nl_layer=nl_layer, gpu_ids=gpu_ids, vaeLike=vaeLike)
    elif which_model_netE == 'conv_512':
        netE = E_NLayers(input_nc, output_nc, ndf, n_layers=6, norm_layer=norm_layer,
                         nl_layer=nl_layer, gpu_ids=gpu_ids, vaeLike=vaeLike)
    elif which_model_netE == 'conv_1024':
        netE = E_NLayers(input_nc, output_nc, ndf, n_layers=7, norm_layer=norm_layer,
                         nl_layer=nl_layer, gpu_ids=gpu_ids, vaeLike=vaeLike)
    else:
        raise NotImplementedError(
            'Encoder model name [%s] is not recognized' % which_model_netE)
    if use_gpu:
        netE.cuda(gpu_ids[0])
    init_weights(netE, init_type=init_type)
    return netE

class ListModule(object):
    # should work with all kind of module
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(
                self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))

class D_NLayers(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3,
                 norm_layer=None, nl_layer=None, use_sigmoid=False, gpu_ids=[]):
        super(D_NLayers, self).__init__()
        self.gpu_ids = gpu_ids

        kw, padw, use_bias = 4, 1, True
        # st()
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw,
                      stride=2, padding=padw, bias=use_bias),
            nl_layer()
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                   kernel_size=kw, stride=2, padding=padw, bias=use_bias)]
            if norm_layer is not None:
                sequence += [norm_layer(ndf * nf_mult)]
            sequence += [nl_layer()]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias)]
        if norm_layer is not None:
            sequence += [norm_layer(ndf * nf_mult)]
        sequence += [nl_layer()]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4,
                               stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        output = self.model(input)
        return output

class D_NLayersMulti(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3,
                 norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[], num_D=1):
        super(D_NLayersMulti, self).__init__()
        # st()
        self.gpu_ids = gpu_ids
        self.num_D = num_D
        if num_D == 1:
            layers = self.get_layers(
                input_nc, ndf, n_layers, norm_layer, use_sigmoid)
            self.model = nn.Sequential(*layers)
        else:
            self.model = ListModule(self, 'model')
            layers = self.get_layers(
                input_nc, ndf, n_layers, norm_layer, use_sigmoid)
            self.model.append(nn.Sequential(*layers))
            self.down = nn.AvgPool2d(3, stride=2, padding=[
                                     1, 1], count_include_pad=False)
            for i in range(num_D - 1):
                ndf = int(round(ndf / (2**(i + 1))))
                layers = self.get_layers(
                    input_nc, ndf, n_layers, norm_layer, use_sigmoid)
                self.model.append(nn.Sequential(*layers))

    def get_layers(self, input_nc, ndf=64, n_layers=3,
                   norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw,
                              stride=2, padding=padw), nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1,
                               kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        return sequence

    def parallel_forward(self, model, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(model, input, self.gpu_ids)
        else:
            return model(input)

    def forward(self, input):
        if self.num_D == 1:
            return self.parallel_forward(self.model, input)
        result = []
        down = input
        for i in range(self.num_D):
            result.append(self.parallel_forward(self.model[i], down))
            if i != self.num_D - 1:
                down = self.parallel_forward(self.down, down)
        return result


# define discriminator
def define_D(input_nc, ndf, which_model_netD='basic_256_multi',
             norm='batch', nl='lrelu',
             use_sigmoid=False, init_type='xavier', num_Ds=1, gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(layer_type=norm)
    nl = 'lrelu'  # use leaky relu for D
    nl_layer = get_non_linearity(layer_type=nl)
    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic_64':
        netD = D_NLayers(input_nc, ndf, n_layers=1, norm_layer=norm_layer,
                         nl_layer=nl_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'basic_128':
        netD = D_NLayers(input_nc, ndf, n_layers=2, norm_layer=norm_layer,
                         nl_layer=nl_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'basic_256':
        netD = D_NLayers(input_nc, ndf, n_layers=3, norm_layer=norm_layer,
                         nl_layer=nl_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'basic_512':
        netD = D_NLayers(input_nc, ndf, n_layers=4, norm_layer=norm_layer,
                         nl_layer=nl_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'basic_1024':
        netD = D_NLayers(input_nc, ndf, n_layers=5, norm_layer=norm_layer,
                         nl_layer=nl_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'basic_64_multi':
        netD = D_NLayersMulti(input_nc=input_nc, ndf=ndf, n_layers=1, norm_layer=norm_layer,
                              use_sigmoid=use_sigmoid, gpu_ids=gpu_ids, num_D=num_Ds)
    elif which_model_netD == 'basic_128_multi':
        netD = D_NLayersMulti(input_nc=input_nc, ndf=ndf, n_layers=2, norm_layer=norm_layer,
                              use_sigmoid=use_sigmoid, gpu_ids=gpu_ids, num_D=num_Ds)
    elif which_model_netD == 'basic_256_multi':
        netD = D_NLayersMulti(input_nc=input_nc, ndf=ndf, n_layers=3, norm_layer=norm_layer,
                              use_sigmoid=use_sigmoid, gpu_ids=gpu_ids, num_D=num_Ds)
    elif which_model_netD == 'basic_512_multi':
        netD = D_NLayersMulti(input_nc=input_nc, ndf=ndf, n_layers=4, norm_layer=norm_layer,
                              use_sigmoid=use_sigmoid, gpu_ids=gpu_ids, num_D=num_Ds)
    elif which_model_netD == 'basic_1024_multi':
        netD = D_NLayersMulti(input_nc=input_nc, ndf=ndf, n_layers=5, norm_layer=norm_layer,
                              use_sigmoid=use_sigmoid, gpu_ids=gpu_ids, num_D=num_Ds)
    else:
        raise NotImplementedError(
            'Discriminator model name [%s] is not recognized' % which_model_netD)
    if use_gpu:
        netD.cuda(gpu_ids[0])
    init_weights(netD, init_type=init_type)
    return netD

# features from vgg19
class VGG_Features(nn.Module):
    def __init__(self):
        super(VGG_Features, self).__init__()
        self.vgg19 = tvmodels.vgg19(pretrained = True).features

    def forward(self, input, select_layers=['13', '22']):
        features = []
        for name, layer in self.vgg19._modules.items():
            input = layer(input)
            if name in select_layers:
                features.append(input)
        return features


def define_VGGF():
    netVGGF = VGG_Features()
    netVGGF.cuda()
    return netVGGF



# some functions
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1.0)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1.0)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)

def get_norm_layer(layer_type='instance'):
    if layer_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif layer_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif layer_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError(
            'normalization layer [%s] is not found' % layer_type)
    return norm_layer

def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(
            nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer

def init_weights(net, init_type='xavier'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [%s] is not implemented' % init_type)

def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=True)


# two usage cases, depend on kw and padw
def upsampleConv(inplanes, outplanes, kw, padw):
    sequence = []
    sequence += [nn.Upsample(scale_factor=2, mode='nearest')]
    sequence += [nn.Conv2d(inplanes, outplanes, kernel_size=kw,
                           stride=1, padding=padw, bias=True)]
    return nn.Sequential(*sequence)


def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [nn.Conv2d(inplanes, outplanes,
                           kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)


def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += [conv3x3(inplanes, outplanes)]
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)


# define some loss function

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, mse_loss=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if mse_loss:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(
                    real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(
                    fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, inputs, target_is_real):
        # if input is a list
        loss = 0.0
        all_losses = []
        for input in inputs:
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss_input = self.loss(input, target_tensor)
            loss = loss + loss_input
            all_losses.append(loss_input)
            # st()
        return loss, all_losses

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class wGANLoss(nn.Module):
    def __init__(self, dir_loss=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(wGANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if dir_loss:
            self.loss = DirLoss()#mean
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(
                    real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(
                    fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, inputs, target_is_real):
        # if input is a list
        loss = 0.0
        all_losses = []
        for input in inputs:
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss_input = self.loss(input, target_tensor)
	    #print loss_input
            loss = loss + loss_input
            #all_losses.append(loss_input)
            # st()
        return loss


# base style loss
class Base_StyleLoss(nn.Module):
    def __init__(self):
        super(Base_StyleLoss, self).__init__()
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def __call__(self, input, target):
        input_gram = self.gram(input)
        target_gram = self.gram(target)
        loss = self.criterion(input_gram, target_gram)
        return loss


# define the style loss
class StyleLoss(nn.Module):
    def __init__(self, vgg_features, select_layers=['0', '2', '5', '7', '10', '12', '14', '16', '19', '21', '23', '25', '28', '30', '32', '34']):
        super(StyleLoss, self).__init__()
        self.VGG_Features = vgg_features
        self.select_layers = select_layers
        self.criterion = Base_StyleLoss()

    def __call__(self, input, target):
        loss = 0.0
        # norm input image for vgg
        # input = self.vgg_norm(input)
        # target = self.vgg_norm(target)
        # input normed image to vgg net
        input_features = self.VGG_Features.forward(input, self.select_layers)
        target_features = self.VGG_Features.forward(target, self.select_layers)
        loss_layer1 = self.criterion(input_features[0].detach(), target_features[0].detach())
        loss_layer2 = self.criterion(input_features[1].detach(), target_features[1].detach())
        loss = loss_layer1 + loss_layer2
        return loss
    def vgg_norm(self,im):
        im.data[:, 0, :, :] = (im.data[:, 0, :, :] * 0.5 + 0.5 - 0.485) / 0.229
        im.data[:, 1, :, :] = (im.data[:, 1, :, :] * 0.5 + 0.5 - 0.456) / 0.224
        im.data[:, 2, :, :] = (im.data[:, 2, :, :] * 0.5 + 0.5 - 0.456) / 0.225
        return im

# Content loss
class ContentLoss(nn.Module):
    def __init__(self, vgg_features, select_layers=['19', '21', '23', '25']):
        super(ContentLoss, self).__init__()
        self.VGG_Features = vgg_features
        self.select_layers = select_layers
        self.criterion = nn.MSELoss()

    def __call__(self, input, target):
        loss = 0.0
        # norm input image for vgg
        #input = self.vgg_norm(input)
        #target = self.vgg_norm(target)
        # input normed image to vgg
        input_features = self.VGG_Features.forward(input, self.select_layers)
        target_features = self.VGG_Features.forward(target, self.select_layers)
        for input_feature, target_feature in zip(input_features, target_features):
            loss = loss + self.criterion(input_feature.detach(), target_feature.detach())
        return loss
    def vgg_norm(self,im):
        im.data[:, 0, :, :] = (im.data[:, 0, :, :] * 0.5 + 0.5 - 0.485) / 0.229
        im.data[:, 1, :, :] = (im.data[:, 1, :, :] * 0.5 + 0.5 - 0.456) / 0.224
        im.data[:, 2, :, :] = (im.data[:, 2, :, :] * 0.5 + 0.5 - 0.456) / 0.225
        return im
    ''' test 
    numpy_target=target_feature.data[0][0].cpu().numpy()
    plt.imshow(numpy_target)
    plt.show()

    numpy_target=input_feature[0][0].data.cpu().numpy()
    plt.imshow(numpy_target)
    plt.show()
    '''

# L2 loss
class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()
        self.criterion = nn.MSELoss()

    def __call__(self, input, target):
        loss = self.criterion(input, target)
        return loss

# mean loss
class DirLoss(nn.Module):
    def __init__(self):
        super(DirLoss, self).__init__()

    def __call__(self, input, target):
        diff = (input - target)
        mean_loss = torch.mean(diff)
        return mean_loss

# gram matrix
class GramMatrix(nn.Module):
    def forward(self, input):
        a, b, c, d = input.size()
        # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

# histogram loss
class HistogramLoss(nn.Module):
    def __init__(self, vgg_features, select_layers=['22']):
        super(HistogramLoss, self).__init__()
        #need vgg feature And selected layers
        self.VGG_Features = vgg_features
        self.select_layers = select_layers
        self.criterion = nn.MSELoss()

    def __call__(self, input, target):
        loss = 0.0
        # norm input image for vgg
        #input = self.vgg_norm(input)
        #target = self.vgg_norm(target)
        #input normed image to vgg
        input_features = self.VGG_Features.forward(input, self.select_layers)
        target_features = self.VGG_Features.forward(target, self.select_layers)
        for input_feature, target_feature in zip(input_features, target_features):
            histmatch_feature = self.histMatch(input_feature,target_feature)
            loss = loss + self.criterion(input_feature.detach(), histmatch_feature.detach())
        return loss

    # this is a coarse version, if the idea works then implement the torch version
    def histMatch(self, imsrc, imtint):
        nbr_bins = 255
        imsrc = self.to_img(imsrc)
        imtint = self.to_img(imtint)
        imres = imsrc.copy()
        for batch_i in range(imsrc.shape[0]):
            for d in range(imsrc.shape[1]):
                # calculate histogram of each image
                imhist, bins = np.histogram(imsrc[batch_i, d, :, :].flatten(), nbr_bins, density=True)
                tinthist, bins = np.histogram(imtint[batch_i, d, :, :].flatten(), nbr_bins, density=True)
                # cumulative distribution function of reference image
                cdfsrc = imhist.cumsum()
                cdfsrc = (255 * cdfsrc / cdfsrc[-1]).astype(np.uint8)  # normalize
                # cumulative distribution function of target image
                cdftint = tinthist.cumsum()
                cdftint = (255 * cdftint / cdftint[-1]).astype(np.uint8)  # normalize
                # use linear interpolation of cdf to find new pixel values
                im2 = np.interp(imsrc[batch_i, d, :, :].flatten(), bins[:-1], cdfsrc)
                im3 = np.interp(im2, cdftint, bins[:-1])
                imres[batch_i, d, :, :] = im3.reshape((imsrc.shape[2], imsrc.shape[3]))
        imres = self.to_torch(imres)
        return imres

    #torch version
    def histogram_torch(self,im,nbr_bins,density=False):
        max = int(im.max())
        min = int(im.min())
        step = float((max-min)/nbr_bins)
        if step!=0:
            bins_t = torch.arange(min, max + step, step)
        else:
            bins_t = torch.zeros(nbr_bins+1)
        imhist_t = torch.histc(im.cpu().float().flatten(), nbr_bins)
        if density==False:
            return imhist_t,bins_t
        else:
            db = step * torch.ones(nbr_bins)
            imhist_t = imhist_t / (imhist_t * db).sum()
            return  imhist_t,bins_t

    def histMatchTorch(self,imsrc,imtint):

        nbr_bins = 255
        #numpy_imsrc
        numpy_imsrc = imsrc.data.cpu().float().numpy()
        imres = numpy_imsrc.copy()
        for batch_i in range(imsrc.shape[0]):
            for d in range(imsrc.shape[1]):
                # calculate histogram of each image
                imhist, bins = self.histogram_torch(imsrc[batch_i, d, :, :].flatten(), nbr_bins, density=True)
                tinthist, bins = self.histogram_torch(imtint[batch_i, d, :, :].flatten(), nbr_bins, density=True)
                # cumulative distribution function of reference image
                cdfsrc = torch.cumsum(imhist,dim=0)
                cdfsrc = (255 * cdfsrc / cdfsrc[-1]).int()  # normalize
                # cumulative distribution function of target image
                cdftint = torch.cumsum(tinthist,dim=0)
                cdftint = (255 * cdftint / cdftint[-1]).int()  # normalize
                # use linear interpolation of cdf to find new pixel values
                im2 = np.interp(imsrc[batch_i, d, :, :].data.flatten(), bins[:-1], cdfsrc)
                im3 = np.interp(im2, cdftint, bins[:-1])
                imres[batch_i, d, :, :] = im3.reshape((imsrc.shape[2], imsrc.shape[3]))

        return torch.from_numpy(imres).cuda()

    def vgg_norm(self,im):
        im.data[:, 0, :, :] = (im.data[:, 0, :, :] * 0.5 + 0.5 - 0.485) / 0.229
        im.data[:, 1, :, :] = (im.data[:, 1, :, :] * 0.5 + 0.5 - 0.456) / 0.224
        im.data[:, 2, :, :] = (im.data[:, 2, :, :] * 0.5 + 0.5 - 0.456) / 0.225
        return im

    def to_img(self,im):
        # return numpy
        pred_im = im.data.cpu().float().numpy()
        pred_im = (pred_im + 1)/2.0 * 255.0
        pred_im = pred_im.astype('uint8')
        return pred_im

    def to_torch(self,im):
        # return numpy
        pre_im = im.astype('float32')
        pre_im = (pre_im/255.0)* 2 - 1
        pre_im = torch.from_numpy(pre_im).cuda()
        pre_im = Variable(pre_im,requires_grad=False)
        return  pre_im

    def to_img_torch(self,im):
        # return torch
        pred_im = (im + 1) / 2.0 * 255.0
        return pred_im.int()
    def to_torch_torch(self,im):
        pred_im = (im/255.0)* 2 - 1
        return pred_im.float()

# feature map
class Feature_map_im(nn.Module):
    def __init__(self, vgg_features, select_layers=['22']):
        super(Feature_map_im, self).__init__()
        self.VGG_Features = vgg_features
        self.select_layers = select_layers

    def __call__(self, input, target):
        # norm input image for vgg
        #input = self.vgg_norm(input)
        #target = self.vgg_norm(target)
        # input normed image to vgg
        input_features = self.VGG_Features.forward(input, self.select_layers)
        target_features = self.VGG_Features.forward(target, self.select_layers)
        return input_features,target_features
    def vgg_norm(self,im):
        im.data[:, 0, :, :] = (im.data[:, 0, :, :] * 0.5 + 0.5 - 0.485) / 0.229
        im.data[:, 1, :, :] = (im.data[:, 1, :, :] * 0.5 + 0.5 - 0.456) / 0.224
        im.data[:, 2, :, :] = (im.data[:, 2, :, :] * 0.5 + 0.5 - 0.456) / 0.225
        return im