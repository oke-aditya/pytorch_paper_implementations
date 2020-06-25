import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

# Squeeze net paper implementation.
# Alexnet like model with 0.5MB model size
# https://arxiv.org/pdf/1602.07360.pdf

# This implementaiton follows torchvision as they use a few more tricks.
