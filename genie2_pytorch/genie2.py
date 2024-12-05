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
        vq_codebook_size = 4096,
        vq_kwargs: dict = dict(),
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

        self.vq = VectorQuantize(
            dim = dim,
            codebook_size = vq_codebook_size,
            rotation_trick = True,
            **vq_kwargs
        )

        self.transformer = nn.Sequential(
            Decoder(
                dim = dim,
                depth = depth,
                heads = heads,
                attn_dim_head = attn_dim_head,
                **transformer_kwargs
            ),
            nn.Linear(dim, dim)
        )

    def forward(
        self,
        state,
        return_loss = False
    ):
        # only need to fold time into batch if not a video enc/dec (classic image enc/dec of today)

        need_fold_time_into_batch = not self.is_video_enc_dec

        if need_fold_time_into_batch:
            state = rearrange(state, 'b c t ... -> b t c ...')
            state, unpack_time = pack_one(state, '* c h w') # state packed into images

        # encode into latents

        latents = self.encoder(state)

        if need_fold_time_into_batch:
            latents = unpack_time(latents, '* c h w')
            latents = rearrange(latents, 'b t c h w -> b c t h w')

        # handle channel first, if encoder does not

        if self.latent_channel_first:
            latents = rearrange(latents, 'b d ... -> b ... d')

        # pack time and spatial fmap into a sequence for transformer

        latents, unpack_time_space_dims = pack_one(latents, 'b * d')

        assert latents.shape[-1] == self.dim_latent

        # project in

        x = self.latent_to_model(latents)

        # discrete quantize - offer continuous later, either using GIVT https://arxiv.org/abs/2312.02116v2 or Kaiming He's https://arxiv.org/abs/2406.11838

        quantized, indices, commit_loss = self.vq(x)

        if return_loss:
            quantized = quantized[:, :-1]
            labels = indices[:, 1:]

        # autoregressive attention

        x = self.transformer(quantized)

        # cross entropy loss off the vq codebook

        if return_loss:
            _, loss = self.vq(
                x,
                indices = labels,
                freeze_codebook = True
            )

            return loss

        # project out

        x = self.model_to_latent(x)

        # restore time and space

        x = unpack_time_space_dims(x)

        if self.latent_channel_first:
            x = rearrange(x, 'b ... d -> b d ...')

        if need_fold_time_into_batch:
            x = rearrange(x, 'b c t h w -> b t c h w')
            x, unpack_time = pack_one(x, '* c h w')

        # decode back to video

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

    loss = genie(x, return_loss = True)
    loss.backward()

    recon = genie(x)
    assert recon.shape == x.shape
