from shutil import rmtree
from pathlib import Path

import torch
from torch import tensor, nn
from torch.nn import Module
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image

rmtree('./results', ignore_errors = True)
results_folder = Path('./results')
results_folder.mkdir(exist_ok = True, parents = True)

# functions

def divisible_by(num, den):
    return (num % den) == 0

# e2e training encoder / decoder / transformer / vq

from x_transformers import Decoder
from vector_quantize_pytorch import VectorQuantize as VQ
from vector_quantize_pytorch.vector_quantize_pytorch import rotate_to

from genie2_pytorch.genie2 import (
    gumbel_sample,
    min_p_filter
)

class Lambda(Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)

class VQImageAutoregressiveAutoencoder(Module):
    def __init__(
        self,
        image_size,
        patch_size,
        dim,
        codebook_size,
        decay = 0.9,
        depth = 3,
        dim_head = 16,
        heads = 4,
        recon_from_pred_codes_weight = 1., # whether to reconstruct from the predicted codes or from the quantized codes directly after vq. 1. means from entirely from predicted code and 0. means entirely right after VQ, so none of the transformer network receives any of the reconstruction gradients
        recon_loss_weight = 1.,
        vq_commit_loss_weight = 1.,
        ar_commit_loss_weight = 1.
    ):
        super().__init__()
        assert divisible_by(image_size, patch_size)

        self.seq_length = (image_size // patch_size) ** 2

        self.encode = nn.Sequential(
            Lambda(lambda x: x * 2 - 1),
            Rearrange('... 1 (h p1) (w p2) -> ...  (h w) (p1 p2)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_size ** 2, dim),
        )

        self.vq = VQ(
            dim = dim,
            codebook_size = codebook_size,
            rotation_trick = True,
            decay = decay
        )

        self.start_token = nn.Parameter(torch.zeros(dim))

        self.decoder = nn.Sequential(
            Decoder(
                dim = dim,
                heads = heads,
                depth = depth,
                attn_dim_head = dim_head,
                rotary_pos_emb = True
            ),
            nn.Linear(dim, dim)
        )

        self.decode = nn.Sequential(
            nn.Linear(dim, patch_size ** 2),
            Rearrange('... (h w) (p1 p2) -> ... 1 (h p1) (w p2)', p1 = patch_size, p2 = patch_size, h = image_size // patch_size),
            Lambda(lambda x: (x + 1) * 0.5),
        )

        self.recon_from_pred_codes_weight = recon_from_pred_codes_weight
        self.recon_loss_weight = recon_loss_weight

        self.vq_commit_loss_weight = vq_commit_loss_weight
        self.ar_commit_loss_weight = ar_commit_loss_weight

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        num_samples = 64,
        min_p = 0.25,
        temperature = 1.5
    ):
        self.eval()

        out = torch.empty((num_samples, 0), dtype = torch.long, device = self.device)

        codebook = self.vq.codebook
        start_tokens = repeat(self.start_token, 'd -> b 1 d', b = num_samples)

        for _ in range(self.seq_length):
            codes = self.vq.get_codes_from_indices(out)

            inp = torch.cat((start_tokens, codes), dim = -2)

            embed = self.decoder(inp)

            logits = -torch.cdist(embed, codebook)

            logits = logits[:, -1]
            logits = min_p_filter(logits, min_p)
            sampled = gumbel_sample(logits, temperature = temperature)

            out = torch.cat((out, sampled), dim = -1)

        sampled_codes = self.vq.get_codes_from_indices(out)
        images = self.decode(sampled_codes)

        return images.clamp(0., 1.)

    def forward(
        self,
        image
    ):
        self.train()

        encoded = self.encode(image)

        quantized, codes, commit_loss = self.vq(encoded)

        # setup autoregressive, patches as tokens scanned from each row left to right

        start_tokens = repeat(self.start_token, '... -> b 1 ...', b = encoded.shape[0])

        tokens = torch.cat((start_tokens, quantized[:, :-1]), dim = -2)

        pred_codes = self.decoder(tokens)

        logits = -torch.cdist(pred_codes, self.vq.codebook)

        ce_loss = F.cross_entropy(
            rearrange(logits, 'b n l -> b l n'),
            codes
        )

        # recon loss, learning autoencoder end to end

        recon_image_from_pred_codes = 0.
        recon_image_from_vq = 0.

        if self.recon_from_pred_codes_weight > 0.:
            rotated_pred_codes = rotate_to(pred_codes, self.vq.get_codes_from_indices(codes))
            recon_image_from_pred_codes = self.decode(rotated_pred_codes)

        if self.recon_from_pred_codes_weight < 1.:
            recon_image_from_vq = self.decode(quantized)

        # weighted combine

        recon_image = (
            recon_image_from_pred_codes * self.recon_from_pred_codes_weight +
            recon_image_from_vq * (1. - self.recon_from_pred_codes_weight)
        )

        # mse loss

        recon_loss = F.mse_loss(
            recon_image,
            image
        )

        # ar commit loss

        ar_commit_loss = F.mse_loss(pred_codes, quantized)

        # total loss and breakdown

        total_loss = (
            ce_loss +
            recon_loss * self.recon_loss_weight +
            commit_loss * self.vq_commit_loss_weight +
            ar_commit_loss * self.ar_commit_loss_weight
        )

        return total_loss, (image, recon_image), (ce_loss, recon_loss, commit_loss, ar_commit_loss)

# model

model = VQImageAutoregressiveAutoencoder(
    dim = 256,
    depth = 4,
    codebook_size = 64,
    decay = 0.95,
    image_size = 28,
    patch_size = 4,
    recon_from_pred_codes_weight = 0.5
)

# data related + optimizer

class MnistDataset(Dataset):
    def __init__(self):
        self.mnist = torchvision.datasets.MNIST(
            './data',
            download = True
        )

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        pil, labels = self.mnist[idx]
        digit_tensor = T.PILToTensor()(pil)
        return (digit_tensor / 255).float()

def cycle(iter_dl):
    while True:
        for batch in iter_dl:
            yield batch

dataset = MnistDataset()

dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)
iter_dl = cycle(dataloader)

optimizer = Adam(model.parameters(), lr = 3e-4)

# train loop

for step in range(1, 100_000 + 1):

    loss, (image, recon_image), (ce_loss, recon_loss, vq_commit_loss, ar_commit_loss) = model(next(iter_dl))
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

    optimizer.step()
    optimizer.zero_grad()    

    loss_str = "\t".join([f"{loss_name}: {loss.item():.3f}"
        for loss_name, loss in (
            ('recon', recon_loss),
            ('ce', ce_loss),
            ('vq commit', vq_commit_loss),
            ('ar commit', ar_commit_loss)
        )
    ])

    print(f'{step}: {loss_str}')

    if divisible_by(step, 500):
        save_image(
            rearrange([image, recon_image], 'ir b 1 h w -> 1 (b h) (ir w)'),
            str(results_folder / f'{step}.train.recon.png')
        )

        image = model.sample(num_samples = 64)

        save_image(
            rearrange(image, '(gh gw) 1 h w -> 1 (gh h) (gw w)', gh = 8).detach().cpu(),
            str(results_folder / f'{step}.png')
        )
