from __future__ import annotations
from beartype import beartype
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

import einx
from einops import rearrange, reduce, pack, unpack

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
        num_actions: int | None = None,
        depth = 12,
        attn_dim_head = 64,
        heads = 8,
        latent_channel_first = False,
        transformer_kwargs: dict = dict(
            add_value_residual = True,
            learned_value_residual_mix = True,
            ff_glu = True,
            use_rmsnorm = True,
        ),
        vq_codebook_size = 4096,
        vq_kwargs: dict = dict(),
        encoder: Module = nn.Identity(),
        decoder: Module = nn.Identity(),
        vq_commit_loss_weight = 1.,
        is_video_enc_dec = False # by default will assume image encoder / decoder, but in the future, video diffusion models with temporal compression will likely perform even better, imo
    ):
        super().__init__()

        self.action_embed = nn.Embedding(num_actions, dim) if exists(num_actions) else None

        self.encoder = encoder
        self.decoder = decoder

        self.is_video_enc_dec = is_video_enc_dec

        self.dim_latent = dim_latent
        self.latent_channel_first = latent_channel_first

        self.latent_to_model = nn.Linear(dim_latent, dim)
        self.model_to_latent = nn.Linear(dim, dim_latent)

        self.vq = VectorQuantize(
            dim = dim_latent,
            codebook_size = vq_codebook_size,
            rotation_trick = False,
            **vq_kwargs
        )

        self.vq_commit_loss_weight = vq_commit_loss_weight

        self.transformer = Decoder(
            dim = dim,
            depth = depth,
            heads = heads,
            attn_dim_head = attn_dim_head,
            **transformer_kwargs
        )

    def forward(
        self,
        state,
        actions = None,
        return_loss = False
    ):

        # handle actions, but allow for state dynamics model to be trained independently

        if exists(actions):
            assert exists(self.action_embed), '`num_actions` must be defined for action embedding on Genie2 before dynamics model can be conditioned on actions'

            assert actions.ndim in {2, 3} # either Int[b, n] or Int[b, n, a] -> for multiple keys being pressed
            actions, _ = pack_one(actions, 'b n *')

            no_actions = actions < 0
            actions = actions.masked_fill(no_actions, 0)

            action_embed = self.action_embed(actions)
            action_embed = einx.where('b n a, b n a d, -> b n a d', ~no_actions, action_embed, 0.)

            action_embed = reduce(action_embed, 'b n a d -> b n d', 'sum')

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

        # discrete quantize - offer continuous later, either using GIVT https://arxiv.org/abs/2312.02116v2 or Kaiming He's https://arxiv.org/abs/2406.11838

        quantized_latents, indices, commit_loss = self.vq(latents)

        if return_loss:
            quantized_latents = quantized_latents[:, :-1]
            labels = indices[:, 1:]

            if exists(actions):
                action_embed = action_embed[:, :-1]

        # project in

        tokens = self.latent_to_model(quantized_latents)

        # add action conditioning, if needed

        if exists(actions):
            tokens = tokens + action_embed

        # autoregressive attention

        tokens = self.transformer(tokens)

        # project out

        tokens = self.model_to_latent(tokens)

        # cross entropy loss off the vq codebook

        if return_loss:
            codebook = self.vq.codebook

            logits = torch.cdist(tokens, codebook)

            state_autoregressive_loss = F.cross_entropy(
                rearrange(logits, 'b n l -> b l n'),
                labels,
                ignore_index = -1
            )

            total_loss = (
                state_autoregressive_loss +
                commit_loss * self.vq_commit_loss_weight
            )

            return total_loss, (state_autoregressive_loss, commit_loss)

        # restore time and space

        tokens = unpack_time_space_dims(tokens)

        if self.latent_channel_first:
            tokens = rearrange(tokens, 'b ... d -> b d ...')

        if need_fold_time_into_batch:
            tokens = rearrange(tokens, 'b c t h w -> b t c h w')
            tokens, unpack_time = pack_one(tokens, '* c h w')

        # decode back to video

        decoded = self.decoder(tokens)

        if need_fold_time_into_batch:
            decoded = unpack_time(decoded, '* c h w')
            decoded = rearrange(decoded, 'b t c h w -> b c t h w')

        return decoded

# quick test

if __name__ == '__main__':
    genie = Genie2(
        dim = 512,
        dim_latent = 768,
        num_actions = 256,
        latent_channel_first = True,
        is_video_enc_dec = True
    )

    x = torch.randn(2, 768, 3, 2, 2)
    actions = torch.randint(0, 256, (2, 12))

    loss, breakdown = genie(x, actions = actions, return_loss = True)
    loss.backward()

    recon = genie(x, actions = actions)
    assert recon.shape == x.shape
