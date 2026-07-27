import torch

from src.ddpm.data.cifar10 import get_cifar10_dataloader
from src.ddpm.models.conditioned_unet import ClassConditionedUNet
from src.ddpm.training.trainer import trainer
from src.ddpm.training.wandb import init_wandb
from src.ddpm.utils.config import load_config
from src.ddpm.utils.seed import set_seed


def main() -> None:
    config = load_config("configs/cifar10.yaml")

    training_config = config["training"]
    dataloader_config = config["dataloader"]
    model_config = config["model"]
    paths_config = config["paths"]
    sampling_config = config["sampling"]

    set_seed(training_config["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataloader = get_cifar10_dataloader(
        data_dir=paths_config["dataset_path"],
        batch_size=dataloader_config["batch_size"],
        num_workers=dataloader_config["num_workers"]
        )

    model = ClassConditionedUNet(
        num_classes=model_config["num_classes"],
        input_dim=model_config["input_dim"],
        channels=model_config["in_channels"],
        groupnorm_groups=model_config["groupnorm_groups"],
        dimension_multiplier=tuple(model_config["dimension_multiplier"]))

    experiment_config = {
        **training_config,
        **dataloader_config,
        **model_config}

    wandb_run = init_wandb(
        wandb_config=config["wandb"],
        experiment_config=experiment_config)

    trainer(
        model=model,
        train_dataloader=train_dataloader,
        device=device,
        save_dir=paths_config["save_path"],
        resume_from=paths_config["resume_from"],
        cifar_val_root=paths_config["cifar_val_path"],
        wandb_run=wandb_run,
        sample_labels=torch.tensor(sampling_config["sample_labels"], dtype=torch.long, device=device),
        **training_config
    )


if __name__ == "__main__":
    main()