from __future__ import annotations

from math import ceil, sqrt
from random import random
from beartype import beartype
from functools import partial

import torch
from torch import nn, tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from torchvision.utils import save_image

import einx
from einops import rearrange, reduce, repeat, pack, unpack

from vector_quantize_pytorch import (
    VectorQuantize,
    ResidualVQ
)

from x_transformers.x_transformers import (
    RotaryEmbedding
)

from x_transformers import (
    Decoder,
    AutoregressiveWrapper
)

from imagen_pytorch import Imagen

# tensor typing

import jaxtyping
from jaxtyping import jaxtyped
from beartype import beartype

class TorchTyping:
    def __init__(self, abstract_dtype):
        self.abstract_dtype = abstract_dtype

    def __getitem__(self, shapes: str):
        return self.abstract_dtype[Tensor, shapes]

Float = TorchTyping(jaxtyping.Float)
Int   = TorchTyping(jaxtyping.Int)
Bool  = TorchTyping(jaxtyping.Bool)

# einstein notation

# b - batch
# c - channels
# t - time
# h - height
# w - width
# n - sequence (flattened latent time * height * width)
# s - space sequence
# l - logits
# a - number of actions (multiple keys pressed)

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def l2norm(t, dim = -1):
    return F.normalize(t, dim = dim, p = 2)

def pack_one(t, pattern):
    packed, ps = pack([t], pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        return unpack(out, ps, inv_pattern)[0]

    return packed, inverse

def project(x, y):
    x, inverse = pack_one(x, 'b *')
    y, _ = pack_one(y, 'b *')

    dtype = x.dtype
    x, y = x.double(), y.double()
    unit = l2norm(y, dim = -1)

    parallel = (x * unit).sum(dim = -1, keepdim = True) * unit
    orthogonal = x - parallel

    return inverse(parallel).to(dtype), inverse(orthogonal).to(dtype)

# input action related helprs

def valid_action_input(inp):
    inp = inp.split(',')
    return all(i.strip().isdigit() for i in inp)

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1, keepdim = True):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim, keepdim = keepdim)

# min_p
# https://arxiv.org/abs/2407.01082

def min_p_filter(logits, min_p = 0.1):
    probs = logits.softmax(dim = -1)
    max_probs = probs.amax(dim = -1, keepdim = True)
    limit = min_p * max_probs
    return torch.where(probs < limit, float('-inf'), logits)

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
        cfg_train_action_dropout = 0.5,
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
        action_autoregressive_loss_weight = 0.1,
        is_video_enc_dec = False # by default will assume image encoder / decoder, but in the future, video diffusion models with temporal compression will likely perform even better, imo
    ):
        super().__init__()

        self.num_actions = num_actions
        self.action_embed = nn.Embedding(num_actions, dim) if exists(num_actions) else None

        self.encoder = encoder
        self.decoder = decoder

        self.is_video_enc_dec = is_video_enc_dec

        self.dim_latent = dim_latent
        self.latent_channel_first = latent_channel_first

        self.latent_to_model = nn.Linear(dim_latent, dim)
        self.model_to_latent = nn.Linear(dim, dim_latent)

        self.time_rotary = RotaryEmbedding(
            dim = attn_dim_head // 2
        )

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

        # behavioral cloning loss weight

        has_action_loss = action_autoregressive_loss_weight > 0.
        self.to_action_pred = nn.Linear(dim, num_actions, bias = False) if has_action_loss else None

        self.has_action_loss = has_action_loss
        self.action_autoregressive_loss_weight = action_autoregressive_loss_weight

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # needed for classifier free guidance

        self.cfg_train_action_dropout = cfg_train_action_dropout

    @torch.no_grad()
    def generate(
        self,
        image: Float['b c h w'],
        num_frames: int,
        filter_kwargs: dict = dict(),
        temperature = 0.9,
        init_action: int | None = None,
        interactive = False,
        interactive_save_last_frame = True
    ):
        was_training = self.training
        self.eval()

        # if interactive is set to True, only allow for sampling one video trajectory at a time

        if interactive:
            assert image.shape[0] == 1
            assert exists(init_action), f'init_action must be given as an integer from 0 - {self.num_actions - 1}'

            actions = tensor([[[init_action]]], device = self.device)
            max_actions = 1

        else:
            actions = None

        # ready image as single frame video

        single_frame = rearrange(image, 'b c h w -> b c 1 h w')

        # encode single frame

        _, first_frame_code, _ = self.encode_state(single_frame)

        # store all latent codes

        space_seq_len = first_frame_code.shape[-1]
        height = int(sqrt(space_seq_len)) # assume square for now

        state_codes = first_frame_code

        # autoregressive sample for number of frames

        for frame in range(1, num_frames + 1):

            if interactive:

                # before prompting the human model, show the human the last image from the world model

                if interactive_save_last_frame:
                    unpacked_codes = rearrange(state_codes, 'b (t h w) -> b t h w', h = height, w = height)
                    last_frame_tokens = self.vq.get_codes_from_indices(unpacked_codes[:, -1])

                    if self.latent_channel_first:
                        last_frame_tokens = rearrange(last_frame_tokens, 'b ... d -> b d ...')

                    last_frame = self.decoder(last_frame_tokens)
                    last_frame = last_frame[0].cpu().detach()
                    channels = last_frame.shape[0]

                    if channels <= 4: # assume valid image type if feature dimension is 4 or less
                        last_frame.clamp_(0., 1.)
                        save_image(last_frame, './last-frame.png')
                    else:
                        torch.save(last_frame, f'./last-frame.pt')

                # prompt human

                while (maybe_next_action := input(f'[frame {frame}] enter the next action (0 - {self.num_actions}): ')) and not valid_action_input(maybe_next_action):
                    print('invalid input, must be integer action - multiple actions need to be all integers separated by commas [ex. "1,3,24"]')

                maybe_next_actions = [*map(int, maybe_next_action.split(','))]
                maybe_next_actions = [*set(maybe_next_actions)]

                next_action = tensor(maybe_next_actions, device = self.device)
                next_action = rearrange(next_action, 'a -> 1 1 a')

                input_num_actions = next_action.shape[-1]

                if input_num_actions > max_actions:
                    actions = F.pad(actions, (0,  input_num_actions - max_actions), value = -1)
                    max_actions = input_num_actions
                elif input_num_actions < max_actions:
                    next_action = F.pad(next_action, (0,  max_actions - input_num_actions), value = -1)

                actions = torch.cat((actions, next_action), dim = 1)

            for _ in range(space_seq_len):

                logits = self.forward(
                    state_codes = state_codes,
                    time_seq_len = frame + 1,
                    actions = actions,
                    return_loss = False,
                )

                last_logit = logits[:, -1]
                last_logit = min_p_filter(last_logit, **filter_kwargs)

                sampled = gumbel_sample(last_logit, temperature = temperature)

                state_codes, _ = pack([state_codes, sampled], 'b *')

        # get all the latent codes

        tokens = self.vq.get_codes_from_indices(state_codes)

        # restore time and space dims

        tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', t = num_frames + 1, h = height)

        if self.latent_channel_first:
            tokens = rearrange(tokens, 'b ... d -> b d ...')

        need_fold_time_into_batch = not self.is_video_enc_dec

        if need_fold_time_into_batch:
            tokens = rearrange(tokens, 'b c t h w -> b t c h w')
            tokens, unpack_time = pack_one(tokens, '* c h w')

        # decode back to video

        video = self.decoder(tokens)

        if need_fold_time_into_batch:
            video = unpack_time(video, '* c h w')
            video = rearrange(video, 'b t c h w -> b c t h w')

        self.train(was_training)

        if not interactive:
            return video

        return video, actions

    def encode_state(
        self,
        state: Float['b c t h w']
    ):

        time_seq_len = state.shape[2]

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

        return self.vq(latents)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        state: Float['b c t h w'] | None = None,
        state_codes: Int['b n'] = None,
        time_seq_len: int | None = None,
        actions: Int['b t'] | Int['b t a'] = None,
        sort_actions = True, # take care of sorting the actions, with any value below 0 as padding
        return_loss = True
    ):
        assert exists(state) ^ exists(state_codes)

        device = self.device

        if not exists(time_seq_len):
            assert exists(state)
            time_seq_len = state.shape[2]

        time_seq = torch.arange(time_seq_len, device = device)

        # handle actions, but allow for state dynamics model to be trained independently

        # when training, adding action embedding depends on the condition dropout probability 

        add_action_embed = (
            exists(actions) and
            (not self.training or random() >= self.cfg_train_action_dropout)
        )

        if add_action_embed:
            assert actions.ndim in {2, 3} # either Int[b, n] or Int[b, n, a] -> for multiple keys being pressed
            assert actions.shape[1] == time_seq_len

            is_multi_action = actions.ndim == 3

            if is_multi_action and sort_actions:
                actions = actions.masked_fill(actions < 0, 1e6)
                actions = actions.sort(dim = -1).values
                actions = actions.masked_fill(actions == 1e6, -1)

            assert exists(self.action_embed), '`num_actions` must be defined for action embedding on Genie2 before dynamics model can be conditioned on actions'

            actions, _ = pack_one(actions, 'b n *')

            no_actions = actions < 0
            actions = actions.masked_fill(no_actions, 0)

            action_embed = self.action_embed(actions)
            action_embed = einx.where('b n a, b n a d, -> b n a d', ~no_actions, action_embed, 0.)

            action_embed = reduce(action_embed, 'b n a d -> b n d', 'sum')

        # encode the state, if state codes are given during sampling, fetch the codes from the vq codebook

        if exists(state):
            quantized_latents, latent_indices, commit_loss = self.encode_state(state)

        elif exists(state_codes):
            latent_indices = state_codes
            quantized_latents = self.vq.get_codes_from_indices(latent_indices)

        # handle rotary positions
        # repeat time across space

        latent_seq_len = quantized_latents.shape[-2]
        spatial_repeat_factor = ceil(latent_seq_len / time_seq_len)

        time_seq = repeat(time_seq, 'n -> (n r)', r = spatial_repeat_factor)

        if add_action_embed:
            action_embed = repeat(action_embed, 'b n d-> b (n r) d', r = spatial_repeat_factor)

        time_rotary_pos = self.time_rotary(time_seq)

        # if returning loss, setup labels for autoregressive loss

        if return_loss:
            quantized_latents = quantized_latents[:, :-1]

            rotary_pos, xpos_scale = time_rotary_pos
            time_rotary_pos = (rotary_pos[:, :-1], xpos_scale)

            labels = latent_indices[:, 1:]

            action_labels = None

            if add_action_embed:
                action_embed = action_embed[:, :-1]

                action_labels = actions[:, 1:]

        # project in

        tokens = self.latent_to_model(quantized_latents)

        tokens_seq_len = tokens.shape[-2]

        # add action conditioning, if needed

        if add_action_embed:
            action_embed = action_embed[:, :tokens_seq_len]

            tokens = tokens + action_embed

        # autoregressive attention

        embed = self.transformer(
            tokens,
            rotary_pos_emb = time_rotary_pos
        )

        # maybe action prediction

        if return_loss and self.has_action_loss:
            is_single_action = actions.ndim == 2 or actions.shape[-1] == 1
            assert is_single_action

            action_time_len = tokens_seq_len // spatial_repeat_factor
            round_down_by_space_len = action_time_len * spatial_repeat_factor
            action_embed = reduce(embed[:, :round_down_by_space_len], 'b (t s) d -> b t d', 'mean', t = action_time_len)

            action_logits = self.to_action_pred(action_embed)

            if actions.ndim == 3:
                actions = rearrange(actions, '... 1 -> ...')

            action_labels = actions[:, 1:]

        # project out

        tokens = self.model_to_latent(embed)

        # cross entropy loss off the vq codebook

        codebook = self.vq.codebook

        logits = -torch.cdist(tokens, codebook)

        if not return_loss:
            return logits

        state_autoregressive_loss = F.cross_entropy(
            rearrange(logits, 'b n l -> b l n'),
            labels,
            ignore_index = -1
        )

        total_loss = (
            state_autoregressive_loss +
            commit_loss * self.vq_commit_loss_weight
        )

        # maybe behavioral cloning

        action_loss = self.zero

        if self.has_action_loss:

            action_loss = F.cross_entropy(
                rearrange(action_logits, 'b n l -> b l n'),
                action_labels,
                ignore_index = -1
            )

            total_loss = (
                total_loss +
                action_loss * self.action_autoregressive_loss_weight
            )

        return total_loss, (state_autoregressive_loss, commit_loss, action_loss)
