import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch import Tensor
import numpy as np
import torch.nn.init as nn_init
from unet_parts import *
from math import sqrt
# import pdb

class GaussianNoise(nn.Module):
    def __init__(self, sigma):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma

    def forward(self, input):
        if self.training:
            noise = Variable(input.data.new(input.size()).normal_(std=self.sigma))
            return input + noise
        else:
            return input

class Expression(nn.Module):
    def __init__(self, func):
        super(Expression, self).__init__()
        self.func = func
    
    def forward(self, input):
        return self.func(input)

class WN_Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, train_scale=False, init_stdv=1.0):
        super(WN_Linear, self).__init__(in_features, out_features, bias=bias)
        if train_scale:
            self.weight_scale = Parameter(torch.ones(self.out_features))
        else:
            self.register_buffer('weight_scale', torch.Tensor(out_features))

        self.train_scale = train_scale 
        self.init_mode = False
        self.init_stdv = init_stdv

        self._reset_parameters()

    def _reset_parameters(self):
        self.weight.data.normal_(0, std=0.05)
        if self.bias is not None:
            self.bias.data.zero_()
        if self.train_scale:
            self.weight_scale.data.fill_(1.)
        else:
            self.weight_scale.fill_(1.)

    def forward(self, input):
        if self.train_scale:
            weight_scale = self.weight_scale
        else:
            weight_scale = Variable(self.weight_scale)

        # normalize weight matrix and linear projection
        # norm_weight = self.weight * (weight_scale.unsqueeze(1) / torch.sqrt((self.weight ** 2).sum(1) + 1e-6)).expand_as(self.weight)
        norm_weight = self.weight * (weight_scale / torch.sqrt((self.weight ** 2).sum(1) + 1e-6)).unsqueeze(1).expand_as(self.weight)
        activation = F.linear(input, norm_weight)

        if self.init_mode == True:
            mean_act = activation.mean(0).squeeze(0)
            activation = activation - mean_act.expand_as(activation)

            inv_stdv = self.init_stdv / torch.sqrt((activation ** 2).mean(0) + 1e-6).squeeze(0)
            activation = activation * inv_stdv.expand_as(activation)

            if self.train_scale:
                self.weight_scale.data = self.weight_scale.data * inv_stdv.data
            else:
                self.weight_scale = self.weight_scale * inv_stdv.data
            self.bias.data = - mean_act.data * inv_stdv.data

        else:
            if self.bias is not None:
                activation = activation + self.bias.expand_as(activation)

        return activation

class WN_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, train_scale=False, init_stdv=1.0):
        super(WN_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if train_scale:
            self.weight_scale = Parameter(torch.Tensor(out_channels))
        else:
            self.register_buffer('weight_scale', torch.Tensor(out_channels))
        
        self.train_scale = train_scale 
        self.init_mode = False
        self.init_stdv = init_stdv

        self._reset_parameters()

    def _reset_parameters(self):
        self.weight.data.normal_(std=0.05)
        if self.bias is not None:
            self.bias.data.zero_()
        if self.train_scale:
            self.weight_scale.data.fill_(1.)
        else:
            self.weight_scale.fill_(1.)

    def forward(self, input):
        if self.train_scale:
            weight_scale = self.weight_scale
        else:
            weight_scale = Variable(self.weight_scale)
        # normalize weight matrix and linear projection [out x in x h x w]
        # for each output dimension, normalize through (in, h, w) = (1, 2, 3) dims
        # norm_weight = self.weight * (weight_scale[:,None,None,None] / torch.sqrt((self.weight ** 2).sum(3).sum(2).sum(1) + 1e-6)).expand_as(self.weight)
        norm_weight = self.weight * (weight_scale / torch.sqrt((self.weight ** 2).sum(3).sum(2).sum(1) + 1e-6))[:,None,None,None].expand_as(self.weight)
        activation = F.conv2d(input, norm_weight, bias=None,
                              stride=self.stride, padding=self.padding, 
                              dilation=self.dilation, groups=self.groups)

        if self.init_mode == True:
            mean_act = activation.mean(3).mean(2).mean(0).squeeze()
            activation = activation - mean_act[None,:,None,None].expand_as(activation)

            inv_stdv = self.init_stdv / torch.sqrt((activation ** 2).mean(3).mean(2).mean(0) + 1e-6).squeeze()
            activation = activation * inv_stdv[None,:,None,None].expand_as(activation)

            if self.train_scale:
                self.weight_scale.data = self.weight_scale.data * inv_stdv.data
            else:
                self.weight_scale = self.weight_scale * inv_stdv.data
            self.bias.data = - mean_act.data * inv_stdv.data

        else:
            if self.bias is not None:
                activation = activation + self.bias[None,:,None,None].expand_as(activation)

        return activation

class WN_ConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, train_scale=False, init_stdv=1.0):
        super(WN_ConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias)
        if train_scale:
            self.weight_scale = Parameter(torch.Tensor(out_channels))
        else:
            self.register_buffer('weight_scale', torch.Tensor(out_channels))
        
        self.train_scale = train_scale 
        self.init_mode = False
        self.init_stdv = init_stdv

        self._reset_parameters()

    def _reset_parameters(self):
        self.weight.data.normal_(std=0.05)
        if self.bias is not None:
            self.bias.data.zero_()
        if self.train_scale:
            self.weight_scale.data.fill_(1.)
        else:
            self.weight_scale.fill_(1.)

    def forward(self, input, output_size=None):
        if self.train_scale:
            weight_scale = self.weight_scale
        else:
            weight_scale = Variable(self.weight_scale)
        # normalize weight matrix and linear projection [in x out x h x w]
        # for each output dimension, normalize through (in, h, w)  = (0, 2, 3) dims
        # norm_weight = self.weight * (weight_scale[None,:,None,None] / torch.sqrt((self.weight ** 2).sum(3).sum(2).sum(0) + 1e-6)).expand_as(self.weight)
        norm_weight = self.weight * (weight_scale / torch.sqrt((self.weight ** 2).sum(3).sum(2).sum(0) + 1e-6))[None,:,None,None].expand_as(self.weight)
        output_padding = self._output_padding(input, output_size)
        activation = F.conv_transpose2d(input, norm_weight, bias=None, 
                                        stride=self.stride, padding=self.padding, 
                                        output_padding=output_padding, groups=self.groups)

        if self.init_mode == True:
            mean_act = activation.mean(3).mean(2).mean(0).squeeze()
            activation = activation - mean_act[None,:,None,None].expand_as(activation)

            inv_stdv = self.init_stdv / torch.sqrt((activation ** 2).mean(3).mean(2).mean(0) + 1e-6).squeeze()
            activation = activation * inv_stdv[None,:,None,None].expand_as(activation)

            if self.train_scale:
                self.weight_scale.data = self.weight_scale.data * inv_stdv.data
            else:
                self.weight_scale = self.weight_scale * inv_stdv.data
            self.bias.data = - mean_act.data * inv_stdv.data

        else:
            if self.bias is not None:
                activation = activation + self.bias[None,:,None,None].expand_as(activation)

        return activation

class thisModule(nn.Module):
    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            elif isinstance(param, Tensor):
                # backwards compatibility for serialized parameters
                param = param
            own_state[name].copy_(param)

class Discriminative(thisModule):
    def __init__(self, config, ema=False):
        super(Discriminative, self).__init__()

        print '===> Init small-conv for {}'.format(config.dataset)

        self.noise_size = config.noise_size
        self.num_label  = config.num_label

        if config.double_input_size:
            self.side = 64
        else:
            self.side = 32
        if config.dataset == 'svhn':
            n_filter_1, n_filter_2 = 64, 128
        elif config.dataset == 'cifar':
            n_filter_1, n_filter_2 = 96, 192
        else:
            raise ValueError('dataset not found: {}'.format(config.dataset))

        drop_ = 0. if hasattr(config, "drop") and config.drop else 0.5
        # Assume X is of size [batch x 3 x 32 x 32]
        self.core_net = nn.Sequential(  # input side size not aware

            nn.Sequential(GaussianNoise(0.05), nn.Dropout2d(0.15)) if config.dataset == 'svhn' \
                else nn.Sequential(GaussianNoise(0.05), nn.Dropout2d(0.2)),

            WN_Conv2d(         3, n_filter_1, 3, 1, 1), nn.LeakyReLU(0.2),
            WN_Conv2d(n_filter_1, n_filter_1, 3, 1, 1), nn.LeakyReLU(0.2),
            WN_Conv2d(n_filter_1, n_filter_1, 3, 2, 1), nn.LeakyReLU(0.2),

            nn.Dropout2d(drop_) if config.dataset == 'svhn' else nn.Dropout(drop_),

            WN_Conv2d(n_filter_1, n_filter_2, 3, 1, 1), nn.LeakyReLU(0.2),
            WN_Conv2d(n_filter_2, n_filter_2, 3, 1, 1), nn.LeakyReLU(0.2),
            WN_Conv2d(n_filter_2, n_filter_2, 3, 2, 1), nn.LeakyReLU(0.2),

            nn.Dropout2d(drop_) if config.dataset == 'svhn' else nn.Dropout(drop_),

            WN_Conv2d(n_filter_2, n_filter_2, 3, 1, 0), nn.LeakyReLU(0.2),
            WN_Conv2d(n_filter_2, n_filter_2, 1, 1, 0), nn.LeakyReLU(0.2),
            WN_Conv2d(n_filter_2, n_filter_2, 1, 1, 0), nn.LeakyReLU(0.2),

            Expression(lambda tensor: tensor.mean(3).mean(2).squeeze()),
        )

        self.out_net = WN_Linear(n_filter_2, self.num_label, train_scale=True, init_stdv=0.1)

        if ema:
            for param in self.parameters():
                param.detach_()

    def forward(self, X, feat=False):
        if X.dim() == 2:
            X = X.view(X.size(0), 3, self.side, self.side)
        
        if feat:
            return self.core_net(X)
        else:
            return self.out_net(self.core_net(X))

class Discriminative_out(thisModule):
    def __init__(self, config):
        super(Discriminative_out, self).__init__()

        print '===> Init small-conv for {}'.format(config.dataset)

        self.num_label  = config.num_label

        if config.dataset == 'svhn':
            n_filter_1, n_filter_2 = 64, 128
        elif config.dataset == 'cifar':
            n_filter_1, n_filter_2 = 96, 192
        else:
            raise ValueError('dataset not found: {}'.format(config.dataset))

        # Assume X is of size [batch x 3 x 32 x 32]

        self.out_net2 = nn.Sequential(
            WN_Linear(n_filter_2, n_filter_2*2, train_scale=True, init_stdv=0.1),
            WN_Linear(n_filter_2*2, self.num_label, train_scale=True, init_stdv=0.1)
        )

        if config.dis_triple:
            self.out_net3 = nn.Sequential(
                WN_Linear(n_filter_2, self.num_label, train_scale=True, init_stdv=0.1)
            )

    def forward(self, X):
            return self.out_net2(X)

def generator(image_size, noise_size=100, large=False, gen_mode='z2i'):
    if gen_mode == 'i2i':
        return UNet(3, 3, large=large, upbilinear=True)
    elif gen_mode == 'z2i':
        return Generator(image_size, noise_size=noise_size, large=large)

class Generator(thisModule):
    def __init__(self, image_size, noise_size=100, large=False):
        super(Generator, self).__init__()

        self.noise_size = noise_size
        self.image_size = image_size

        if not large:   # side: 32
            self.core_net = nn.Sequential(
                nn.Linear(self.noise_size, 4 * 4 * 512, bias=False), nn.BatchNorm1d(4 * 4 * 512), nn.ReLU(), 
                Expression(lambda tensor: tensor.view(tensor.size(0), 512, 4, 4)),
                nn.ConvTranspose2d(512, 256, 5, 2, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
                nn.ConvTranspose2d(256, 128, 5, 2, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
                WN_ConvTranspose2d(128,   3, 5, 2, 2, 1, train_scale=True, init_stdv=0.1), nn.Tanh(),
            )
        else:   # side: 64
            self.core_net = nn.Sequential(
                nn.Linear(self.noise_size, 2 * 2 * 1024, bias=False), nn.BatchNorm1d(2 * 2 * 1024), nn.ReLU(), 
                Expression(lambda tensor: tensor.view(tensor.size(0), 1024, 2, 2)),
                nn.ConvTranspose2d(1024, 512, 5, 2, 2, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(),
                nn.ConvTranspose2d(512, 256, 5, 2, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
                nn.ConvTranspose2d(256, 256, 5, 2, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
                nn.ConvTranspose2d(256, 128, 5, 2, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
                WN_ConvTranspose2d(128,   3, 5, 1, 2, 0, train_scale=True, init_stdv=0.1), nn.Tanh(),
            )

    def forward(self, noise):        
        output = self.core_net(noise)

        return output

class Encoder(thisModule):
    def __init__(self, image_size, noise_size=100, output_params=False):
        super(Encoder, self).__init__()

        self.noise_size = noise_size
        self.image_size = image_size

        self.core_net = nn.Sequential(
            nn.Conv2d(  3, 128, 5, 2, 2, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 5, 2, 2, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, 5, 2, 2, bias=False), nn.BatchNorm2d(512), nn.ReLU(),
            Expression(lambda tensor: tensor.view(tensor.size(0), -1)), # 512 * 4 * 4)),
        )
        
        if output_params:
            self.core_net.add_module(str(len(self.core_net._modules)), WN_Linear(4 * 4 * 512, self.noise_size*2, train_scale=True, init_stdv=0.1))
            self.core_net.add_module(str(len(self.core_net._modules)), Expression(lambda x: torch.chunk(x, 2, 1)))
        else:
            self.core_net.add_module(str(len(self.core_net._modules)), WN_Linear(4 * 4 * 512, self.noise_size, train_scale=True, init_stdv=0.1))

    def forward(self, input):
        
        output = self.core_net(input)

        return output

    def load_my_state_dict(self, state_dict):

        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            if isinstance(param, Tensor):
                # backwards compatibility for serialized parameters
                param = param
            own_state[name].copy_(param)

class UNet(thisModule):
    def __init__(self, n_channels, n_classes, large=False, upbilinear=True):
        # n_channels: input channels; n_classes: output channels
        super(UNet, self).__init__()    # cifar: 64 => 16
        if large:
            cnum = 32   # min channel num
        else:
            cnum = 64   # min channel num
        self.inc = inconv(n_channels, cnum)
        self.down1 = down(cnum,   cnum*2)
        self.down2 = down(cnum*2, cnum*4)
        self.down3 = down(cnum*4, cnum*8)
        self.down4 = down(cnum*8, cnum*8)
        self.up1 = up(cnum*16, cnum*4, bilinear=upbilinear)
        self.up2 = up(cnum*8,  cnum*2, bilinear=upbilinear)
        self.up3 = up(cnum*4,  cnum, bilinear=upbilinear)
        self.up4 = up(cnum*2,  cnum, bilinear=upbilinear)
        self.outc = WN_Conv2d(cnum, n_classes, 1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, x):
        x = x * 0.5 + 0.5  # [-1, 1] -> [0, 1]
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = F.sigmoid(x) * 2. - 1. # [0, 1] -> [-1, 1]
        return x

'''
class Discriminative2(thisModule):
    def __init__(self, config, ema=False,
                 n_filter_1=96, n_filter_2=192,
                 block_config=(3), bn_size=4, drop_rate=0):
        super(Discriminative2, self).__init__()

        print '===> Init small-conv for {}'.format(config.dataset)

        self.noise_size = config.noise_size
        self.num_label  = config.num_label

        if config.double_input_size:
            self.side = 64
        else:
            self.side = 32
        # if config.dataset == 'svhn':
        #     n_filter_1, n_filter_2 = 64, 128
        # elif config.dataset == 'cifar':
        #     n_filter_1, n_filter_2 = 96, 192
        # else:
        #     raise ValueError('dataset not found: {}'.format(config.dataset))

        drop_ = 0. if hasattr(config, "drop") and config.drop else 0.5
        # Assume X is of size [batch x 3 x 32 x 32]
        self.core_net = nn.Sequential(  # input side size not aware

            nn.Sequential(GaussianNoise(0.05), nn.Dropout2d(0.15)) if config.dataset == 'svhn' \
                else nn.Sequential(GaussianNoise(0.05), nn.Dropout2d(0.2)),

            WN_Conv2d(         3, n_filter_1, 3, 1, 1), nn.LeakyReLU(0.2),
            WN_Conv2d(n_filter_1, n_filter_1, 3, 1, 1), nn.LeakyReLU(0.2),
            WN_Conv2d(n_filter_1, n_filter_1, 3, 2, 1), nn.LeakyReLU(0.2),

            nn.Dropout2d(drop_) if config.dataset == 'svhn' else nn.Dropout(drop_),

            WN_Conv2d(n_filter_1, n_filter_2, 3, 1, 1), nn.LeakyReLU(0.2),
            WN_Conv2d(n_filter_2, n_filter_2, 3, 1, 1), nn.LeakyReLU(0.2),
            WN_Conv2d(n_filter_2, n_filter_2, 3, 2, 1), nn.LeakyReLU(0.2),

            nn.Dropout2d(drop_) if config.dataset == 'svhn' else nn.Dropout(drop_),

            WN_Conv2d(n_filter_2, n_filter_2, 3, 1, 0), nn.LeakyReLU(0.2),
            WN_Conv2d(n_filter_2, n_filter_2, 1, 1, 0), nn.LeakyReLU(0.2),
            WN_Conv2d(n_filter_2, n_filter_2, 1, 1, 0), nn.LeakyReLU(0.2),

            # Expression(lambda tensor: tensor.mean(3).mean(2).squeeze()),
        )
        self.dense_net = nn.Sequential()
        num_features = n_filter_2
        for i, num_layers in enumerate(block_config):
            growth_rate = n_filter_2 // num_layers  # 32
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.dense_net.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
            self.dense_net.add_module('transition%d' % (i + 1), trans)
            num_features = num_features // 2
        self.out_net = WN_Linear(num_features, self.num_label, train_scale=True, init_stdv=0.1)

        if ema:
            for param in self.parameters():
                param.detach_()
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, X, feat=False):
        if X.dim() == 2:
            X = X.view(X.size(0), 3, self.side, self.side)

        X = self.core_net(X)
        X = self.dense_net(X)
        if feat:
            return X
        else:
            return self.out_net(X)
'''