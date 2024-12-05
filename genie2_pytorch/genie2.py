from __future__ import annotations
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

import einx
from einops import rearrange

from x_transformers import (
    Attention
)

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

# main class

class Genie2(Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, x):
        return x
