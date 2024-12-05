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
        decoder: Module = nn.Identity(),
        is_video_enc_dec = False # by default will assume image encoder / decoder, but in the future, video diffusion models with temporal compression will likely perform even better, imo
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.is_video_enc_dec = is_video_enc_dec

        self.dim_latent = dim_latent
        self.latent_channel_first = latent_channel_first

        self.latent_to_model = nn.Linear(dim_latent, dim)
        self.model_to_latent = nn.Linear(dim, dim_latent)

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
        need_fold_time_into_batch = self.is_video_enc_dec

        if need_fold_time_into_batch:
            state = rearrange(state, 'b c t ... -> b t c ...')
            state, unpack_time = pack_one(state, '* c h w') # state packed into images

        latents = self.encoder(state)

        if need_fold_time_into_batch:
            latents = unpack_time(latents, '* c h w')
            latents = rearrange(latents, 'b t c h w -> b c t h w')

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

        if need_fold_time_into_batch:
            x = rearrange(x, 'b c t h w -> b t c h w')
            x, unpack_time = pack_one(x, '* c h w')

        decoded = self.decoder(x)

        if need_fold_time_into_batch:
            decoded = unpack_time(decoded, '* c h w')
            decoded = rearrange(decoded, 'b t c h w -> b c t h w')

        return decoded

# quick test

if __name__ == '__main__':
    genie = Genie2(
        dim = 512,
        dim_latent = 768,
        latent_channel_first = True,
        is_video_enc_dec = True
    )

    x = torch.randn(1, 768, 3, 2, 2)

    out = genie(x)
