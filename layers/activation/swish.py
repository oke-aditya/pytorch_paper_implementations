# Adapted from rwightman
import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(x, inplace: bool = False):
    """ Swish activation function as described in https://arxiv.org/pdf/1710.05941.pdf
    """
    if inplace:
        return x.mul_(x.sigmoid())
    else:
        return x.mul(x.sigmoid())

class Swish(nn.Module):
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return swish(x, inplace=self.inplace)

# if __name__ == "__main__":
#     x = torch.tensor(3, dtype=torch.float)
#     nr2 = Swish(inplace=True)
#     print(nr2.forward(x))

#     x = torch.tensor(3, dtype=torch.float)
#     print(nr2(x))
