# DDPM PyTorch
![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/License-MIT-green)

A PyTorch implementation of Denoising Diffusion Probabilistic Models (DDPM) trained on CIFAR-10.
The project started as an implementation of the original DDPM paper and then became a combination of ideas from multiple diffusion works, making it suitable for experimentation and further extension.

The model now uses the following features:
- DDPM sampling
- DDIM sampling
- Velocity prediction
- Class-conditioned generation
- Classifier-Free Guidance (CFG)
- Cosine noise schedule
- Sinusoidal time embeddings
- Exponential Moving Average (EMA)
- YAML configuration
- Checkpointing and training resumption

## Installation
To download and install the repository run the following commands:
```
git clone https://github.com/RickyPyeet/ddpm-pytorch.git
cd ddpm-pytorch
pip install -r requirements.txt
```

## Training
Training is configured, and can be modified, via `configs/cifar10.yaml`. It can be performed by launching:

>`python train.py`

When training is launched CIFAR-10 is downloaded automatically.

## Sampling
When sampling, follow these steps:
1. Download one of the available checkpoints from [Hugging Face](https://huggingface.co/Pitto16/DDPM/tree/main)
2. Place the checkpoint in `checkpoints/`
3. Run the following command: `python sample.py --checkpoint checkpoints/*specific_checkpoint*.pt --sampler ddim  --labels 0 1 2 3 4 5 6 7`

To save generated samples:

```
python sample.py --checkpoint checkpoints/ddpm_small.pt --sampler ddim --labels 0 1 2 3 --save_img --save_path outputs/
```

## References
- [Ho et al., Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Song et al., Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
- [Ho & Salimans, Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
- [Nichol & Dhariwal, Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)
- [Lai et al., The principles of Diffusion Models](https://arxiv.org/abs/2510.21890)





