#!/usr/bin/env python

import os
import sys
import warnings
from typing import List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import DictConfig
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import Logger
from lightning.pytorch.callbacks import Callback
from mattermake.utils.pylogger import get_pylogger

warnings.filterwarnings(
    "ignore", message=".*fractional coordinates rounded to ideal values.*"
)

log = get_pylogger(__name__)


def train(config: DictConfig) -> Optional[float]:
    """Trains the crystal transformer model. Can additionally evaluate on a test set.

    Args:
        config: Configuration composed by Hydra.

    Returns:
        Optional[float]: Validation loss if training succeeded.
    """

    if config.get("seed"):
        log.info(f"Setting random seed: {config.seed}")
        seed_everything(config.seed, workers=True)

    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if cb_conf is not None and "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    loggers: List[Logger] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if lg_conf is not None and "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                loggers.append(hydra.utils.instantiate(lg_conf))

    log.info(f"Instantiating data module <{config.data._target_}>")
    data_module = hydra.utils.instantiate(config.data)
    data_module.prepare_data()
    data_module.setup(stage="fit")

    tokenizer = data_module.get_tokenizer()
    vocab_size = tokenizer.vocab_size if tokenizer else config.model.vocab_size

    config.model.vocab_size = vocab_size

    log.info(f"Instantiating model <{config.model._target_}>")
    model = hydra.utils.instantiate(config.model)

    if tokenizer:
        model.tokenizer_config = {
            "idx_to_token": tokenizer.idx_to_token,
            "token_to_idx": tokenizer.vocab,
            "lattice_bins": tokenizer.lattice_bins,
            "coordinate_precision": tokenizer.coordinate_precision,
        }
        
    # Configure ground truth hook for continuous regression losses
    if config.model.prediction_mode == "continuous":
        log.info("Setting up continuous regression loss hooks for continuous mode")
        
        # Set ground truth values from batch data during training
        def prepare_ground_truth_hook(pl_module, batch):
            # This hook extracts ground truth values from the batch data
            # and sets them in the model for regression loss calculation
            if hasattr(pl_module.model, "set_ground_truth_values"):
                # Extract lattice parameters and coordinates from the batch
                # This requires adapting your dataloader to provide this information
                if "ground_truth_lattice" in batch:
                    pl_module.model.set_ground_truth_values(
                        lattice_params=batch["ground_truth_lattice"],
                        fractional_coords=batch.get("ground_truth_coords")
                    )
                    
        # Register the hook with the model
        model.on_train_batch_start = prepare_ground_truth_hook

    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    ckpt_path = config.get("ckpt_path")
    if ckpt_path and os.path.exists(ckpt_path):
        log.info(f"Loading checkpoint: {ckpt_path}")

    if config.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=data_module, ckpt_path=ckpt_path)

    if config.get("test"):
        log.info("Starting testing!")
        trainer.test(model=model, datamodule=data_module, ckpt_path="best")

    try:
        return trainer.callback_metrics["val_loss"].item()
    except Exception:
        return None


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="train_crystal_transformer",
)
def main(config: DictConfig) -> Optional[float]:
    if config.get("debug"):
        log.info("Running in debug mode!")
        config.trainer.fast_dev_run = True

    return train(config)


if __name__ == "__main__":
    main()
