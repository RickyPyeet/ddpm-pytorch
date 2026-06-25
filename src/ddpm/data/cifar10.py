import torch
from torch import torchvision
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

def get_train_transforms():
  train_transform = transforms.Compose([
      transforms.Resize((32, 32)),
      transforms.RandomHorizontalFlip(p = 0.5),
      transforms.ToTensor(),
      transforms.Lambda(lambda t: (t*2) - 1)
      ])
  return train_transform

def get_cifar10_dataloader(data_dir: str,
                           batch_size: int,
                           num_workers: int = 4,
                           shuffle: bool = True):
  train_dataset = CIFAR10(root = data_dir,
                          train = True,
                          transform = get_train_transforms(),
                          dowload = True)
  
  train_dataloader = DataLoader(dataset = train_dataset,
                                batch_size = batch_size,
                                shuffle = shuffle,
                                num_workers = num_workers,
                                pin_memory = True)
  return train_dataloader
