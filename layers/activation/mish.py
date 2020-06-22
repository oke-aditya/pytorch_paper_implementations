# Mish implementation
# Taken from Mish author himself https://github.com/digantamisra98/Mish/
# And from Rwigthman working style

import torch
import torch.nn as nn
import torch.nn.functional as F

def mish(x, inplace: bool = False):
    """
    Applies the mish function element-wise:
    Mish: A Self Regularized Non-Monotonic Neural Activation Function - https://arxiv.org/abs/1908.08681
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    """
    return x.mul(F.tanh(F.softplus(x)))

class Mish(nn.Module):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
    
    def forward(self, x):
        return mish(x)



    
