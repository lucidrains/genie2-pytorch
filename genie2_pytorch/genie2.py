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
        dim_latent,
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

        self.latent_to_model = nn.Linear(dim, dim_latent)
        self.model_to_latent = nn.Linear(dim_latent, dim)

        self.transformer = Decoder(
            dim = dim,
            depth = depth,
            heads = heads,
            attn_dim_head = attn_dim_head,
            **transformer_kwargs
        )

    def forward(
        self,
        state
    ):
        latents = self.encoder(state)

        x = self.latent_to_model(latents)
        x = self.transformer(x)
        x = self.model_to_latent(x)

        decoded = self.decoder(x)

        return decoded
