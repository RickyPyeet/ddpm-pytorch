import wandb

def init_wandb(wandb_config, experiment_config, run_id = None):
  if not wandb_config.get('use_wandb', False):
    return None

  pred_type = experiment_config['pred_type']
  seed = experiment_config['seed']

  run_name = f"pred_type_{pred_type}_seed_{seed}"

  run = wandb.init(entity = wandb_config.get('entity'),
                   project = wandb_config['project'],
                   name = run_name,
                   group = wandb_config.get('group'),
                   job_type = wandb_config.get('job_type', 'train'),
                   config = experiment_config,
                   mode = wandb_config.get('mode', 'online'),
                   id = run_id,
                   resume = 'allow')

  return run
