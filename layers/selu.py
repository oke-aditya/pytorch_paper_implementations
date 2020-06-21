# This is a raw selu implementation.
# You can use torch.nn.SeLU simply in your code.
# This is plain simple implementation made by me.
# Learn more from here https://towardsdatascience.com/gentle-introduction-to-selus-b19943068cd9

import torch
from torch.nn import functional as F
import torch.nn as nn

def selu(x, inplace: bool = False):
    """ Selu as described in the paper Self-Normalizing Neural Networks https://arxiv.org/pdf/1706.02515.pdf """
    # Values taken from the paper
    alpha = 1.67326324
    scale = 1.05070098
    if inplace:
        if(x < 0):
            x.mul_(scale * alpha * (x.exp_() - 1))
        else:
            x.mul_(scale)
        return x

    else:
        if(x < 0):
            x.mul(scale * alpha * (x.exp() - 1))
        else:
            x.mul(scale)
        return x


class Selu(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
    
    def forward(self, x):
        return(selu(x, self.inplace))

# Sample code to check it

# if __name__ == "__main__":
#     x = torch.tensor(3, dtype=torch.float)
#     nr2 = Selu(inplace=True)
#     print(nr2.forward(x))


#     x = torch.tensor(3, dtype=torch.float)
#     print(nr2(x))



