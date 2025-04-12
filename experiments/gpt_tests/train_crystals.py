import torch
import hydra
import rootutils
import lightning.pytorch as pl
from omegaconf import DictConfig
from lightning.pytorch import Trainer, LightningModule, Callback, LightningDataModule
from lightning.pytorch.loggers import Logger

from mattermake import utils
from mattermake.utils.distributed_init import (
    configure_pytorch,
    init_distributed_mode,
    log_distributed_settings,
    patch_lightning_slurm_master_addr,
)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = utils.pylogger.get_pylogger(__name__)


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="train_crystals_vae.yaml"
)
def main(cfg: DictConfig):
    """Train the crystal diffusion model.

    Args:
        cfg: Configuration composed by Hydra.

    Returns:
        Tuple with metrics and dict with all instantiated objects.
    """
    # Set seed for reproducibility
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    if (
        isinstance(cfg.trainer.get("strategy"), str)
        and cfg.trainer.get("strategy") == "ddp"
    ):
        cfg.trainer.strategy = {
            "_target_": "lightning.pytorch.strategies.DDPStrategy",
            "find_unused_parameters": True,
        }

    log.info(f"Instantiating data module <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    loggers: list[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=loggers
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": loggers,
        "trainer": trainer,
    }

    if loggers:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("compile"):
        log.info("Compiling model!")
        model = torch.compile(model)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    test_metrics = trainer.callback_metrics

    # Merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


if __name__ == "__main__":
    pl.seed_everything(42)
    patch_lightning_slurm_master_addr()
    init_distributed_mode(port=12354)
    log_distributed_settings(log)
    configure_pytorch(log)
    main()
