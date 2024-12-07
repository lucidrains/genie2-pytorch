<img src="./genie2.png" width="400px"></img>

## Genie2 - Pytorch (wip)

Implementation of a framework for <a href="https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/">Genie2</a> in Pytorch

## Install

```bash
$ pip install genie2-pytorch
```

## Usage

```python
import torch
from genie2_pytorch import Genie2

genie = Genie2(
    dim = 512,
    depth = 12,
    dim_latent = 768,
    num_actions = 256,
    latent_channel_first = True,
    is_video_enc_dec = True
)

video = torch.randn(2, 768, 3, 2, 2)
actions = torch.randint(0, 256, (2, 3))

loss, breakdown = genie(video, actions = actions)
loss.backward()

generated_video = genie.generate(video[:, :, 0], num_frames = 16)

assert generated_video.shape == (2, 768, 16 + 1, 2, 2)
```

## Citations

```bibtex
@inproceedings{Valevski2024DiffusionMA,
    title   = {Diffusion Models Are Real-Time Game Engines},
    author  = {Dani Valevski and Yaniv Leviathan and Moab Arar and Shlomi Fruchter},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:271962839}
}
```

```bibtex
@article{Ding2024DiffusionWM,
    title     = {Diffusion World Model},
    author    = {Zihan Ding and Amy Zhang and Yuandong Tian and Qinqing Zheng},
    journal   = {ArXiv},
    year      = {2024},
    volume    = {abs/2402.03570},
    url       = {https://api.semanticscholar.org/CorpusID:267499902}
}
```

```bibtex
@inproceedings{Sadat2024EliminatingOA,
    title   = {Eliminating Oversaturation and Artifacts of High Guidance Scales in Diffusion Models},
    author  = {Seyedmorteza Sadat and Otmar Hilliges and Romann M. Weber},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:273098845}
}
```

```bibtex
@misc{ParkerHolder2024,
    author  = {Jack Parker-Holder, Philip Ball, Jake Bruce, Vibhavari Dasagi, Kristian Holsheimer, Christos Kaplanis, Alexandre Moufarek, Guy Scully, Jeremy Shar, Jimmy Shi, Stephen Spencer, Jessica Yung, Michael Dennis, Sultan Kenjeyev, Shangbang Long, Vlad Mnih, Harris Chan, Maxime Gazeau, Bonnie Li, Fabio Pardo, Luyu Wang, Lei Zhang, Frederic Besse, Tim Harley, Anna Mitenkova, Jane Wang, Jeff Clune, Demis Hassabis, Raia Hadsell, Adrian Bolton, Satinder Singh, Tim Rocktäschel},
    url     = {https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/}
}
```
