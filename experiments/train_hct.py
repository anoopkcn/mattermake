from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import torch
from lightning import Callback, Trainer
from omegaconf import DictConfig
import rootutils

from mattermake.models.hct_module import HierarchicalCrystalTransformer
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
    # Set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # Set precision for numerical operations
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: HierarchicalCrystalTransformer = hydra.utils.instantiate(cfg.model)

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

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        if not cfg.get("train") or cfg.trainer.get("fast_dev_run"):
            # Load best checkpoint after training
            ckpt_path = cfg.get("ckpt_path")
            if not ckpt_path:
                log.warning(
                    "Best checkpoint not found! Using current model parameters for testing..."
                )
                ckpt_path = None
        else:
            ckpt_path = "best"

        test_metrics = trainer.test(
            model=model, datamodule=datamodule, ckpt_path=ckpt_path
        )

        # Update metrics with test metrics
        if test_metrics and len(test_metrics) > 0:
            metrics = test_metrics[0]
        else:
            metrics = {}

        # Combine with training metrics
        metrics.update(train_metrics)
    else:
        metrics = train_metrics

    # Return metric dictionary and object dictionary for later use
    return metrics, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_hct.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # Apply optional utilities
    extras(cfg)

    # Train model
    metric_dict, _ = train(cfg)

    # Return optimized metric value for optuna (if applicable)
    optimized_metric = cfg.get("optimized_metric")
    if optimized_metric:
        return get_metric_value(metric_dict, optimized_metric)


if __name__ == "__main__":
    main()
