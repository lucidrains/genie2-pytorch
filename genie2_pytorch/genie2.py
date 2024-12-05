from __future__ import annotations
from beartype import beartype
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

import einx
from einops import rearrange, pack, unpack

from vector_quantize_pytorch import (
    VectorQuantize,
    ResidualVQ
)

from x_transformers import (
    Decoder,
    AutoregressiveWrapper
)

from imagen_pytorch import Imagen

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def pack_one(t, pattern):
    packed, ps = pack([t], pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        return unpack(out, ps, inv_pattern)[0]

    return packed, inverse

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
        latent_channel_first = False,
        transformer_kwargs: dict = dict(),
        encoder: Module = nn.Identity(),
        decoder: Module = nn.Identity()
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.dim_latent = dim_latent
        self.latent_channel_first = latent_channel_first

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

        if self.latent_channel_first:
            latents = rearrange(latents, 'b d ... -> b ... d')

        latents, unpack_time_space_dims = pack_one(latents, 'b * d')

        assert latents.shape[-1] == self.dim_latent

        x = self.latent_to_model(latents)
        x = self.transformer(x)
        x = self.model_to_latent(x)

        x = unpack_time_space_dims(x)

        if self.latent_channel_first:
            x = rearrange(x, 'b ... d -> b d ...')

        decoded = self.decoder(x)

        return decoded
