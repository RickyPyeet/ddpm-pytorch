import torch
from torch import nn

from ddpm.diffusion.process import batched_diffusion_kernel
from ddpm.diffusion.schedules import DiffusionSchedule
from ddpm.sampling.inference import generate_and_plot
from ddpm.training.ema import EMA
from ddpm.training.objective import get_train_target
from ddpm.utils.checkpoint import load_checkpoint, save_checkpoint



def trainer(model: nn.Module,
            train_dataloader,
            epochs,
            device,
            pred_type: str = 'v',
            lr: float = 3e-4,
            class_free_dropout: float = 0.2,
            guidance_scale: float = 2.5,
            eta: float = 0.8,
            ema_decay: float = 0.999,
            timesteps: int = 1000,
            optim: str = "adamw",
            save_dir: str = None,
            save_every: int = 200,
            resume_from: str = None,
            sample_every: int | None = 50,
            sample_labels: torch.Tensor | None = None,
            sampler: str = 'ddpm',
            sample_timesteps: int = 100,
            use_compile: bool = False,
            seed: int = 42):
  
  valid_pred_type =['epsilon', 'x_0', 'v']

  if pred_type not in valid_pred_type:
    raise ValueError(f"Pred_type must be one of {valid_pred_type}, got {pred_type} instead")

  model = model.to(device)

  # Create optimizer
  optimizer = create_optimizer(model = model,
                               optim_type = optim,
                               lr = lr)

  # Create loss_fn
  loss_fn = torch.nn.MSELoss()

  # EMA
  ema = EMA(model = model, decay = ema_decay)

  # Noise Schedule
  schedule = DiffusionSchedule(timesteps = timesteps, schedule = 'cosine', device = device)
  betas, alphas, alpha_bar = schedule()

  # Define training variables
  starting_epoch = 0
  loss_hist = []
  checkpoint = {}

  # Load checkpoint if exists
  if resume_from is not None:
    checkpoint = load_checkpoint(checkpoint_path = resume_from,
                                 model = model,
                                 optimizer = optimizer,
                                 ema = ema,
                                 device = device)

    starting_epoch = checkpoint['epoch']+1
    pred_type = checkpoint['pred_type']
    loss_hist = checkpoint['loss_hist']

    class_free_dropout = checkpoint['class_free_dropout']
    guidance_scale = checkpoint['guidance_scale']

  # Compile model for faster training
  if use_compile:
    torch.set_float32_matmul_precision('high')
    model.compile()

  for epoch in tqdm(range(starting_epoch, epochs)):
    print(f"Epoch {epoch+1}/{epochs}\n------------------")
    model.train()
    train_loss = 0.0

    for images, labels in train_dataloader:
      images = images.to(device)
      labels = labels.to(device)

      batch_size = images.shape[0]

      # Extract random noise steps between 0-1000
      t = torch.randint(0, timesteps, (batch_size, ), device = device, dtype = torch.long)

      ### FORWARD DIFFUSION
      noisy_images, noise = batched_diffusion_kernel(x_not = images,
                                                     t = t,
                                                     alpha_bars = alpha_bar)

      ### Drop Mask for CFG
      drop_mask = torch.rand(batch_size, device = device) < class_free_dropout
      c_input = labels.clone()
      c_input[drop_mask] = -1 # null class token is identified with '-1'

      # Make a prediction
      pred_target = model(noisy_images, time = t, c = c_input)

      # Extract target based on pred_type
      target = get_train_target(x_0 = images,
                                noise = noise,
                                alpha_bar = alpha_bar,
                                timestep = t,
                                pred_type = pred_type)

      # Compute loss and optimize
      loss = loss_fn(pred_target, target)
      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 10.0)
      optimizer.step()
      ema.update()

      train_loss += loss.item()

    train_loss /= len(train_dataloader)
    loss_hist.append(train_loss)

    print(f"Training Loss: {train_loss:.5f}\n")

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'epoch': epoch,
        'pred_type': pred_type,
        'loss_hist': loss_hist,
        'class_free_dropout': class_free_dropout
    }
    # Save checkpoints
    if save_dir is not None and ((epoch+1) % save_every == 0) or ((epoch+1) == epochs):
      save_checkpoint(name = f"checkpoint_epoch_{epoch+1}_{lr}_lr_{pred_type}_pred_type.pt",
                      checkpoint = checkpoint,
                      checkpoint_path = save_dir)
    
    # Generate samples
    if sample_every is not None:
      if (epoch + 1) % sample_every == 0:
        ema.apply_shadow()
        model.eval()

        with torch.inference_mode():
          if sample_labels is None:
            sample_labels = torch.tensor([1, 3, 6, 8])
          generate_and_plot(model = model,
                            c = sample_labels,
                            img_shape = (len(sample_labels), 3, 32, 32),
                            sampler = sampler,
                            timesteps = timesteps,
                            sampling_timesteps = sample_timesteps,
                            pred_type = pred_type,
                            guidance_scale = guidance_scale,
                            eta = eta,
                            title = f"Epoch: {epoch}",
                            seed = seed)

        ema.restore()

  return loss_hist, checkpoint
