from __future__ import annotations
from beartype import beartype
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

import einx
from einops import rearrange

from vector_quantize_pytorch import (
    VectorQuantize,
    ResidualVQ
)

from imagen_pytorch import Imagen

from x_transformers import (
    Decoder
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
    @beartype
    def __init__(
        self,
        dim,
        depth = 12,
        attn_dim_head = 64,
        heads = 8,
        transformer_kwargs: dict = dict(),
        encoder: Module = nn.Identity(),
        decoder: Module = nn.Identity()
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.transformer = Decoder(
            dim = dim,
            depth = depth,
            heads = heads,
            attn_dim_head = dim_head,
            **transformer_kwargs
        )

    def forward(
        self,
        state
    ):
        encoded = self.encoder(state)

        attended = self.transformer(encoded)

        decoded = self.decoder(attended)

        return decoded
