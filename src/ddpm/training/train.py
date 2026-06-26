import torch
from torch import nn
from ddpm.utils.seed import set_seed
from ddpm.utils.config import load_config
from ddpm.data.cifar10 import get_cifar10_dataloader
from ddpm.models.conditioned_unet import ClassConditionedUNet
from ddpm.training.trainer import train

def main():

  config = load_config('configs/cifar10.yaml')

  set_seed(config['seed'])
  
  device = config['device']
  if device == 'cuda' and not torch.cuda.is_available():
    device = 'cpu'

  train_dataloader = get_cifar10_dataloader(data_dir = config['dataset']['root'],
                                            batch_size = config['dataset']['batch_size'],
                                            num_workers = config['dataset']['num_workers'])
  
  model = ClassConditionedUNet(num_classes = config['model']['num_classes'],
                               input_dim = config['model']['input_dim'],
                               channels = config['model']['channels'],
                               groupnorm_groups = config['model']['groupnorm_groups'],
                               dimension_multiplier = tuple(config['model']['dimension_multiplier']))

  trainer(model=model,
          train_dataloader=train_dataloader,
          epochs=config['training']['epochs'],
          device=device,
          pred_type=config['diffusion']['prediction_type'],
          lr=config['training']['learning_rate'],
          class_free_dropout=config['cfg']['dropout'],
          guidance_scale=config['cfg']['guidance_scale'],
          eta=config['sampling']['eta'],
          ema_decay=config['training']['ema_decay'],
          timesteps=config['diffusion']['timesteps'],
          schedule_type=config['diffusion']['schedule'],
          optim=config['training']['optimizer'],
          save_dir=config['training']['save_dir'],
          save_every=config['training']['save_every'],
          resume_from=config['training']['resume_from'],
          sample_every=config['sampling']['sample_every'],
          sample_labels=torch.tensor(config['sampling']['labels']),
          sampler=config['sampling']['sampler'],
          sample_timesteps=config['sampling']['sampling_timesteps'],
          use_compile=config['training']['use_compile'])

if __name__ == '__main__':
  main()
