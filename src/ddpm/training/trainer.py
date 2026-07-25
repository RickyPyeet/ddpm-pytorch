import torch
from torch import nn
from tqdm.auto import tqdm

from src.ddpm.diffusion.process import batched_diffusion_kernel
from src.ddpm.diffusion.schedules import DiffusionSchedule
from src.ddpm.sampling.inference import generate_and_plot
from src.ddpm.evaluating.fid import generate_compute_fid
from src.ddpm.training.ema import EMA
from src.ddpm.training.objective import get_train_target
from src.ddpm.utils.checkpoint import load_checkpoint, save_checkpoint

def create_optimizer(model, optim_type, lr):
  optim_list = ['adam', 'adamw']

  if optim_type not in optim_list:
    raise ValueError(f"{optim_type} is not a valid optimizer, pick {optim_list}")
  
  if optim_type == 'adam':
    optimizer = torch.optim.Adam(params = model.parameters(),
                                 lr = lr)
  elif optim_type == 'adamw':
    optimizer = torch.optim.AdamW(params = model.parameters(),
                                  lr = lr,
                                  betas = (0.9, 0.999),
                                  eps = 1e-8,
                                  weight_decay = 0.01)
  return optimizer

# Crate scheduler
def create_cosine_scheduler(optimizer, T_max):
  """
  Create a learning rate cosine scheduler
  """
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer, T_max = T_max)
  return scheduler

# Trainer function
def trainer(model: nn.Module,
            train_dataloader,
            device,
            save_dir: str = None,
            resume_from: str = None,
            cifar_val_root: str = None,
            wandb_run = None,
            sample_labels: torch.Tensor | None = None,
            total_steps: int = 500000,
            pred_type: str = 'v',
            lr: float = 3e-4,
            use_amp: bool = True,
            scaler_dtype: str = 'float16',
            log_every: int = 100,
            save_every: int = 200,
            sample_every: int | None = 50,
            timesteps: int = 1000,
            schedule_type: str = 'cosine',
            optim: str = "adamw",
            ema_decay: float = 0.999,
            class_free_dropout: float = 0.2,
            seed: int = 42,
            guidance_scale: float = 2.5,
            eta: float = 0.8,
            sampler: str = 'ddpm',
            sample_timesteps: int = 100,
            enable_fid: bool = False, 
            num_classes: int = 10,
            fid_every: int = 100000,
            num_samples: int = 10000,
            fid_batch_size: int = 256,
            fid_sampler: str = 'ddim'):

  if seed is not None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)

  valid_pred_type =['epsilon', 'x_0', 'v']

  if pred_type not in valid_pred_type:
    raise ValueError(f"Pred_type must be one of {valid_pred_type}, got {pred_type} instead")

  model = model.to(device)

  # Create optimizer
  optimizer = create_optimizer(model = model,
                               optim_type = optim,
                               lr = lr)

  # Create scaler for AMP and the datatype
  scaler = torch.amp.GradScaler('cuda', enabled = use_amp)

  if scaler_dtype == 'float16':
    autocast_dtype = torch.float16
  elif scaler_dtype == 'bfloat16':
    autocast_dtype = torch.bfloat16
  else:
    raise ValueError(f"Scaler data type {scaler_dtype} must be [float16, bfloat16]")

  # Create lr scheduler
  scheduler = create_cosine_scheduler(optimizer, total_steps)

  # Create loss_fn
  loss_fn = torch.nn.MSELoss()

  # EMA
  ema = EMA(model = model, decay = ema_decay)

  # Noise Schedule
  schedule = DiffusionSchedule(timesteps = timesteps, schedule = schedule_type, device = device)
  betas, alphas, alpha_bar = schedule()

  # Define training variables
  steps = 0
  running_loss = 0.0
  running_grad_norm = 0.0
  loss_hist = []
  grad_norm_hist = []
  checkpoint = {}
  fid_hist = []
  best_fid = float('inf')

  # Load checkpoint if exists
  if resume_from is not None:
    checkpoint = load_checkpoint(checkpoint_path = resume_from,
                                 model = model,
                                 optimizer = optimizer,
                                 ema = ema,
                                 scheduler = scheduler,
                                 scaler = scaler,
                                 device = device)

    steps = checkpoint['steps']
    pred_type = checkpoint['pred_type']
    loss_hist = checkpoint['loss_hist']
    grad_norm_hist = checkpoint['grad_norm_hist']
    fid_hist = checkpoint['fid_hist']
    best_fid = checkpoint['best_fid']

    class_free_dropout = checkpoint['class_free_dropout']

  # Setup tqdm
  tq = tqdm(desc = 'Training steps', total = total_steps, initial = steps)

  # Train loop
  while steps < total_steps:
    print(f"Steps: {steps}/{total_steps}\n------------------")
    model.train()

    for images, labels in train_dataloader:
      images = images.to(device)
      labels = labels.to(device)

      batch_size = images.shape[0]

      # Extract random noise steps between 0-999
      t = torch.randint(0, timesteps, (batch_size, ), device = device, dtype = torch.long)

      ### FORWARD DIFFUSION
      noisy_images, noise = batched_diffusion_kernel(x_not = images,
                                                     t = t,
                                                     alpha_bars = alpha_bar)

      ### Drop Mask for CFG
      drop_mask = torch.rand(batch_size, device = device) < class_free_dropout
      c_input = labels.clone()
      c_input[drop_mask] = -1 # null class token is identified with '-1'


      # Extract the target
      target = get_train_target(x_0 = images,
                                noise = noise,
                                alpha_bar = alpha_bar,
                                timestep = t,
                                pred_type = pred_type)

      # Use autocast and make prediction and calculate loss
      with torch.autocast(device_type = device, dtype = autocast_dtype, enabled = use_amp):
        # Make a prediction
        pred_target = model(noisy_images, time = t, c = c_input)
        # Compute loss and optimize
        loss = loss_fn(pred_target, target)


      # Update scaler
      scaler.scale(loss).backward()
      scaler.unscale_(optimizer)

      # Clip grad norm
      grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 10.0)

      # Scaler step and update
      scaler.step(optimizer)
      scaler.update()

      # Step the scheduler
      scheduler.step()

      # Zero optimizer
      optimizer.zero_grad()

      # Update EMA
      ema.update()

      # Update loss and grad_norm
      running_loss += loss.item()
      running_grad_norm += grad_norm.item()

      # Increase steps counter and tqdm
      steps += 1
      tq.update(1)

      # Log metrics and print
      if steps % log_every == 0:
        avg_loss = running_loss / log_every
        avg_grad_norm = running_grad_norm / log_every
        print(f"Training Loss: {avg_loss:.5f} | Grad Norm: {avg_grad_norm:.5f}\n")
        running_loss = 0.0
        running_grad_norm = 0.0

        loss_hist.append(avg_loss)
        grad_norm_hist.append(avg_grad_norm)

        # Log wandb run
        if wandb_run is not None:
          wandb_run.log({'train_loss': avg_loss,
                        'gradient_norm': avg_grad_norm,
                        'lr': optimizer.param_groups[0]['lr']},
                        step = steps)


      # Define FID related variables
      fid_results = None
      is_new_best_fid = False

      # Generate samples
      if sample_every is not None and steps % sample_every == 0:
        ema.apply_shadow()
        model.eval()

        with torch.inference_mode():
          if sample_labels is None:
            sample_labels = torch.tensor([0,1,2,3,4,5,6,7,8,9])
          generate_and_plot(model = model,
                            c = sample_labels,
                            img_shape = (len(sample_labels), 3, 32, 32),
                            sampler = sampler,
                            timesteps = timesteps,
                            sampling_timesteps = sample_timesteps,
                            pred_type = pred_type,
                            guidance_scale = guidance_scale,
                            eta = eta,
                            title = f"Steps: {steps}",
                            seed = seed)

        ema.restore()
        model.train()

      # Compute FID if required
      if enable_fid and steps % fid_every == 0:
        model.eval()
        print(f"Start computing FID...\n")
        fid, fid_results = generate_compute_fid(cifar_val_root = cifar_val_root,
                                                num_samples = num_samples,
                                                batch_size = fid_batch_size,
                                                num_classes = num_classes,
                                                model = model,
                                                timesteps = timesteps,
                                                sampler = fid_sampler,
                                                ema = ema,
                                                sampling_timesteps = sample_timesteps,
                                                pred_type = pred_type,
                                                guidance_scale = guidance_scale,
                                                eta = eta)
        model.train()
        fid_hist.append(fid)

        print(f"FID: {fid:.4f} | Sampling time: {fid_results['sampling_time_seconds']:.3f} s | ms per image: {fid_results['ms_per_image']:.4f}\n")

        if wandb_run is not None:
          wandb_run.log({'fid': fid,
                         'fid/sampling_time_seconds': fid_results['sampling_time_seconds'],
                         'fid/ms_per_image': fid_results['ms_per_image'],
                         'fid/fid_time_seconds': fid_results['fid_time']},
                        step = steps)

        # Check best FID and update
        if fid < best_fid:
          best_fid = fid
          is_new_best_fid = True

      # Bool checks for saving
      should_save_periodic = save_dir is not None and (steps % save_every == 0 or steps == total_steps)
      should_save_best = save_dir is not None and is_new_best_fid

      # Build checkpoint
      if should_save_periodic or should_save_best:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'ema_state_dict': ema.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'steps': steps,
            'pred_type': pred_type,
            'loss_hist': loss_hist,
            'grad_norm_hist': grad_norm_hist,
            'class_free_dropout': class_free_dropout,
            'wandb_run_id': wandb_run.id if wandb_run is not None else None,
            'fid_hist': fid_hist,
            'best_fid': best_fid
        }
        if fid_results is not None:
          checkpoint['fid_results'] = fid_results

      # Save checkpoints
      if should_save_periodic:
        save_checkpoint(name = f"checkpoint_steps_{steps}_{lr}_lr_{pred_type}_pred_type.pt",
                        checkpoint = checkpoint,
                        checkpoint_path = save_dir)

      # Log FID and keep track of best one
      if should_save_best:
        save_checkpoint(name = f"best_fid_steps_{steps}_{lr}_lr_{pred_type}_pred_type.pt",
                        checkpoint = checkpoint,
                        checkpoint_path = save_dir)

      # Check if total_steps were reached and exit loop if true
      if steps >= total_steps:
        break

  # Finish the run
  if wandb_run is not None:
    wandb_run.finish()

  return loss_hist, grad_norm_hist, checkpoint