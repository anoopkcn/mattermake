from typing import Any, Dict, List, Optional, Tuple
import os
import hydra

import lightning.pytorch as pl
from lightning import Callback, Trainer
from omegaconf import DictConfig
import rootutils

from mattermake.utils.distributed_init import (
    configure_pytorch,
    init_distributed_mode,
    log_distributed_settings,
    patch_lightning_slurm_master_addr,
)

from mattermake.models.modular_hierarchical_crystal_transformer_module import (
    ModularHierarchicalCrystalTransformer,
)
from mattermake.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Trains the Hierarchical Crystal Transformer model. Can additionally evaluate on a test set and compute metrics.

        :param cfg: A DictConfig configuration composed by Hydra.
        :return: A tuple with metrics dict and the dict with all instantiated objects.
    """
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: ModularHierarchicalCrystalTransformer = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    loggers = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=loggers
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "trainer": trainer,
        "callbacks": callbacks,
        "logger": loggers,
    }

    if loggers:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    metrics = trainer.callback_metrics

    return metrics, object_dict


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="train_modular_hct.yaml"
)
def main(cfg: DictConfig) -> Optional[float]:
    extras(cfg)

    metric_dict, _ = train(cfg)

    optimized_metric = cfg.get("optimized_metric")
    if optimized_metric:
        return get_metric_value(metric_dict, optimized_metric)


if __name__ == "__main__":
    pl.seed_everything(42)
    patch_lightning_slurm_master_addr()
    if os.environ.get("SLURM_NTASKS"):
        init_distributed_mode(port=12354)
    log_distributed_settings(log)
    configure_pytorch(log)
    main()
