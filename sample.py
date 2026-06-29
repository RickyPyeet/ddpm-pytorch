import torch
import argparse
from src.ddpm.utils.config import load_config
from src.ddpm.utils.seed import set_seed
from src.ddpm.models.conditioned_unet import ClassConditionedUNet
from src.ddpm.training.ema import EMA
from src.ddpm.sampling.inference import generate_and_plot

def parse_arg():
  parser = argparse.ArgumentParser()

  # Arguments
  parser.add_argument('--checkpoint', type = str, required = True, help = 'Model checkpoint path')
  parser.add_argument('--config', type = str, default = 'configs/cifar10.yaml', help = 'Model config path')
  parser.add_argument('--sampler', type = str, default = 'ddim', choices = ['ddpm', 'ddim'])
  parser.add_argument('--labels', type = int, nargs = '+', default = [1,3,6,8], help = 'labels to generate')
  parser.add_argument('--save_path', type = str, default = None)
  parser.add_argument('--save_img', action = 'store_true')
  parser.add_argument('--seed', type = int, default = 42)

  return parser.parse_args()

def main():
  args = parse_arg()

  config = load_config(args.config)
  set_seed(args.seed)

  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  ckpt = torch.load(f = args.checkpoint, map_location = device)

  model = ClassConditionedUNet(num_classes = config['model']['num_classes'],
                               input_dim = config['model']['input_dim'],
                               channels = config['model']['channels'],
                               groupnorm_groups = config['model']['groupnorm_groups'],
                               dimension_multiplier = tuple(config['model']['dimension_multiplier'])).to(device)
  
  model.load_state_dict(state_dict = ckpt['model_state_dict'])

  if 'ema_state_dict' in ckpt:
    ema = EMA(model = model, decay = 0.999)
    ema.load_state_dict(state_dict = ckpt['ema_state_dict'])
    ema.apply_shadow()

  labels = torch.tensor(args.labels, dtype = torch.long, device = device)
  with torch.inference_mode():
    generate_and_plot(model = model,
                      c = labels,
                      img_shape = ((len(labels),config['model']['channels'],32,32)),
                      sampler =  args.sampler,
                      timesteps = config['diffusion']['timesteps'],
                      sampling_timesteps = config['sampling']['sampling_timesteps'],
                      pred_type = config['diffusion']['prediction_type'],
                      guidance_scale = config['cfg']['guidance_scale'],
                      eta = config['sampling']['eta'],
                      save_img = args.save_img,
                      save_path = args.save_path,
                      seed = args.seed)

if __name__ == '__main__':
  main()