import time
import os

import torch
from torch import nn
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from src.ddpm.sampling.ddpm import sample_ddpm
from src.ddpm.sampling.ddim import sample_ddim

from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from tempfile import TemporaryDirectory

def save_samples(samples: torch.Tensor,
                 save_path: str | Path,
                 start_from: int = 0) -> None:
  """
  Takes a batch samples and saves them in '.png' format.
  The image will be saved with the following structure:
  save_path/00000.png
  """
  save_path = Path(save_path)
  save_path.mkdir(parents = True, exist_ok = True)

  batch_size = samples.shape[0]

  pil_images = tensor_to_pil(samples)

  for i, img in enumerate(pil_images):
    img_path = save_path / f"{start_from+i:06d}.png"
    img.save(img_path)

  print(f"Finished saving {batch_size} images to {save_path}!")

def generate_samples(num_samples: int,
                     batch_size: int,
                     num_classes: int,
                     save_path: str | Path,
                     model: nn.Module,
                     timesteps: int,
                     sampling_timesteps: int,
                     sampler: str = 'ddim',
                     ema = None,
                     pred_type: str = 'v',
                     guidance_scale: float = 3.5,
                     eta: float = 0.8):

  device = next(model.parameters()).device

  # Check if sampler used is correct
  if sampler not in ['ddpm', 'ddim']:
    raise ValueError(f"Invalid sampler {sampler}, use [ddpm, ddim]")

  # Create class labels based on amount of samples we need to create
  all_labels = torch.arange(num_samples, dtype = torch.long, device = device) % num_classes

  # Shuffle labels
  permutation = torch.randperm(num_samples, device = device)
  all_labels = all_labels[permutation]

  # Eval mode
  model.eval()

  if ema is not None:
    ema.apply_shadow()

  generated_count = 0
  sampling_time_seconds = 0.0

  with torch.inference_mode():
    for i in tqdm(range(0, num_samples, batch_size), desc = "Generating samples"):
      # Compute end_idx for leftover values
      end_idx = min(i+batch_size, num_samples)

      # compute current batch of labels
      current_batch_size = end_idx - i
      labels = all_labels[i:end_idx]

      # Start measuring time
      if torch.cuda.is_available():
        torch.cuda.synchronize()

      batch_start = time.perf_counter()

      if sampler == 'ddim':
        samples, _ = sample_ddim(model = model,
                                timesteps = timesteps,
                                sampling_timesteps = sampling_timesteps,
                                img_shape = (current_batch_size,  3, 32, 32),
                                c = labels,
                                pred_type = pred_type,
                                save_steps = False,
                                guidance_scale = guidance_scale,
                                eta = eta)
      elif sampler == 'ddpm':
        samples, _ = sample_ddpm(model = model,
                                 timesteps = timesteps,
                                 img_shape = (current_batch_size,  3, 32, 32),
                                 c = labels,
                                 pred_type = pred_type,
                                 save_steps = False,
                                 guidance_scale = guidance_scale)
      # End batch generating time
      if torch.cuda.is_available():
        torch.cuda.synchronize()
      sampling_time_seconds += time.perf_counter() - batch_start

      # Save samples
      print(f"Saving samples in {save_path}... \n")
      save_samples(samples = samples,
                   save_path = save_path,
                   start_from = i)

      generated_count += current_batch_size

  if ema is not None:
    ema.restore()

  # Compute total elapsed time and time metrics
  samples_per_seconds = generated_count / sampling_time_seconds

  ms_per_image = sampling_time_seconds / generated_count * 1000


  print(f"Finished sampling!\n")
  print(f"Generated {generated_count} images in {sampling_time_seconds:.2f} | {samples_per_seconds} images/s")

  return sampling_time_seconds, samples_per_seconds, ms_per_image, generated_count

class GeneratedDataset(Dataset):
  def __init__(self, folder: str | Path):
    self.files = sorted(Path(folder).glob('*.png'))

    if not self.files:
      raise ValueError(f"No PNG images found in {folder}")

  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx):
    image = Image.open(self.files[idx]).convert("RGB")
    return transforms.PILToTensor()(image)


@torch.inference_mode()
def compute_fid(generated_dir: str | Path,
                cifar_root: str | Path,
                batch_size: int = 256,
                num_workers: int = 2,
                device = None):
  if device is None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

  # Start timer
  if torch.cuda.is_available():
    torch.cuda.synchronize()
  start = time.perf_counter()

  real_dataset = CIFAR10(root = cifar_root,
                         train = False,
                         download = True,
                         transform = transforms.PILToTensor())

  generated_dataset = GeneratedDataset(generated_dir)

  real_dataloader = DataLoader(dataset = real_dataset,
                               batch_size = batch_size,
                               shuffle = False,
                               num_workers = num_workers,
                               pin_memory = device == 'cuda')

  generated_dataloader = DataLoader(dataset = generated_dataset,
                                batch_size = batch_size,
                                shuffle = False,
                                num_workers = num_workers,
                                pin_memory = device == 'cuda')

  fid = FrechetInceptionDistance(feature = 2048, normalize = False).to(device)

  for real_img, _ in real_dataloader:
    real_img = real_img.to(device)
    fid.update(real_img, real = True)

  for gen_img in generated_dataloader:
    gen_img = gen_img.to(device)
    fid.update(gen_img, real = False)

  score = fid.compute()

  # End timer
  if torch.cuda.is_available():
    torch.cuda.synchronize()
  elapsed_time = time.perf_counter() - start

  return score.item(), elapsed_time

def generate_compute_fid(cifar_val_root: str | Path,
                        num_samples: int,
                        batch_size: int,
                        num_classes:int,
                        model: nn.Module,
                        timesteps: int,
                        sampler: str,
                        ema,
                        sampling_timesteps: int,
                        pred_type: str,
                        guidance_scale: float,
                        eta: float):

device = next(model.parameters()).device

with TemporaryDirectory() as tmpdir:
sampling_time_seconds, samples_per_second, ms_per_image, generated_count = generate_samples(num_samples = num_samples,
                                                                                            batch_size = batch_size,
                                                                                            num_classes = num_classes,
                                                                                            save_path = tmpdir,
                                                                                            model = model,
                                                                                            timesteps = timesteps,
                                                                                            sampler = sampler,
                                                                                            ema = ema,
                                                                                            sampling_timesteps = sampling_timesteps,
                                                                                            pred_type = pred_type,
                                                                                            guidance_scale = guidance_scale,
                                                                                            eta = eta)

fid, fid_time = compute_fid(generated_dir = tmpdir,
                            cifar_root = cifar_val_root,
                            batch_size = batch_size,
                            num_workers = os.cpu_count(),
                            device = device)

# Create results dictionary
results = {'fid': fid,
            'fid_time': fid_time,
            'generated_count': generated_count,
            'sampling_time_seconds': sampling_time_seconds,
            'samples_per_second': samples_per_second,
            'ms_per_image': ms_per_image}

return fid, results