# Orignal GANs were not Convolutional Networks. They were simple FFN.
# Now-a-days when we talk about GANs its usually DCGAN.
# This implemenation follows 
# https://github.com/pytorch/examples/blob/master/dcgan/main.py
# Which is very close to the orignal paper https://arxiv.org/pdf/1511.06434.pdf

# Some guideliness which the paper suggested.
# Architecture guidelines for stable Deep Convolutional GANs
# •Replace any pooling layers with strided convolutions (discriminator) and fractional-stridedconvolutions (generator).
# •Use batchnorm in both the generator and the discriminator.
# •Remove fully connected hidden layers for deeper architectures.
# •Use ReLU activation in generator for all layers except for the output, which uses Tanh.
# •Use LeakyReLU activation in the discriminator for all layers.

# This is just implemenation of DCGAN network. Here I'm not illustrating DataLoading and Data Handling

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import config
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# custom weights initialization called on Generator and Discriminator

def weights_init(m):
    classname = m.__class___.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight, 1.0, 0.02)
        init.zeros_(m.bias)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),

            # State size. (ngf * 8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
    
    def forward(self, x):
        out = self.gen(x)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        out = self.disc(x)
        return out.view(-1, 1).squeeze(1)

