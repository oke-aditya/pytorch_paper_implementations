""" 
Taken from Ross Wightman

PyTorch selectable adaptive pooling
Adaptive pooling with the ability to select the type of pooling from:
    * 'avg' - Average pooling
    * 'max' - Max pooling
    * 'avgmax' - Sum of average and max pooling re-scaled by 0.5
    * 'avgmaxc' - Concatenation of average and max pooling along feature dim, doubles feature dim
Both a functional and a nn.Module version of the pooling is provided.
Author: Ross Wightman (rwightman)

"""

import torch.nn as nn
import torch
import torch.nn.functional as F

__all__ = ["selective_adaptive_pool2d", "SelectiveAdaptivePool2d"]


def adpative_pool_feat_mult(pool_type="avg"):
    if pool_type == "catavgmax":
        return 2
    else:
        return 1


def adaptive_avgmax_pool2d(x, output_size=1):
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return 0.5 * (x_avg + x_max)


def adaptive_catavgmax_pool2d(x, output_size=1):
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return torch.cat((x_avg, x_max), 1)


def selective_adaptive_pool2d(x, pool_type="avg", output_size=1):
    """ Selective global pooling with dynamic kernel size """
    if pool_type == "avg":
        x = F.adaptive_avg_pool2d(x, output_size)
    elif pool_type == "max":
        x = F.adaptive_max_pool2d(x, output_size)
    elif pool_type == "avgmax":
        x = adaptive_avgmax_pool2d(x, output_size)
    elif pool_type == "catavgmax":
        x = adaptive_catavgmax_pool2d(x, output_size)
    else:
        assert False, "Invalid Pool type %s" % (pool_type)

    return x


class AdaptiveAvgMaxPool2d(nn.Module):
    def __init__(self, output_size=1):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_avgmax_pool2d(x, 1)


class AdaptiveCatAvgMaxPool2d(nn.Module):
    def __init__(self, output_size=1):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_catavgmax_pool2d(x, self.output_size)


class SelectiveAdaptivePool2d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size
    """

    def __init__(self, output_size=1, pool_type="avg", flatten=False):
        super().__init__()
        self.output_size = output_size
        self.pool_type = pool_type
        self.flatten = flatten
        if pool_type == "avgmax":
            self.pool = AdaptiveAvgMaxPool2d(output_size)
        elif pool_type == "catavgmax":
            self.pool = AdaptiveCatAvgMaxPool2d(output_size)
        elif pool_type == "avg":
            self.pool = nn.AdaptiveAvgPool2d(output_size)
        elif pool_type == "max":
            self.pool = nn.AdaptiveMaxPool2d(output_size)
        else:
            if pool_type != "avg":
                assert False, "Invalid pool type: %s" % pool_type
            self.pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        x = self.pool(x)
        if self.flatten:
            return x.flatten(1)
        return x

    def feat_mult(self):
        return adpative_pool_feat_mult(self.pool_type)

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + "output_size="
            + str(self.output_size)
            + ", pool_type="
            + self.pool_type
            + ")"
        )
