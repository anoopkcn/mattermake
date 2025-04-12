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
from mattermake.models.hierarchical_crystal_transformer_module import (
    HierarchicalCrystalTransformerModule,
)

warnings.filterwarnings(
    "ignore", message=".*fractional coordinates rounded to ideal values.*"
)

log = get_pylogger(__name__)


class EnhancedCrystalTransformerModule(HierarchicalCrystalTransformerModule):
    """Enhanced version of HierarchicalCrystalTransformerModule that ensures
    tokenizer configuration is properly saved and loaded with checkpoints.
    Also supports KV-caching for generation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_kv_cache_for_eval = False

    def on_save_checkpoint(self, checkpoint):
        # Call the parent class implementation first
        super().on_save_checkpoint(checkpoint)

        # Save tokenizer config in checkpoint
        if hasattr(self, "tokenizer_config") and self.tokenizer_config is not None:
            checkpoint["tokenizer_config"] = self.tokenizer_config
            log.info("Saved tokenizer config in checkpoint")

    def on_load_checkpoint(self, checkpoint):
        # Call the parent class implementation first
        super().on_load_checkpoint(checkpoint)

        # Load tokenizer config from checkpoint
        if "tokenizer_config" in checkpoint:
            self.tokenizer_config = checkpoint["tokenizer_config"]
            log.info("Loaded tokenizer config from checkpoint")

    def enable_kv_cache_for_evaluation(self, enable=True):
        """Enable KV-caching for model evaluation and generation"""
        self.use_kv_cache_for_eval = enable
        log.info(f"KV-caching for evaluation {'enabled' if enable else 'disabled'}")

        # Keep reference to original method for patching
        if not hasattr(self, "_original_generate_structure"):
            self._original_generate_structure = self.generate_structure

            # Create patched version that uses KV-caching
            def generate_with_kv_cache(*args, **kwargs):
                if self.use_kv_cache_for_eval:
                    kwargs["use_kv_cache"] = True
                    log.info("Using KV-caching for structure generation")
                return self._original_generate_structure(*args, **kwargs)

            # Replace the method
            self.generate_structure = generate_with_kv_cache


def train(config: DictConfig) -> Optional[float]:
    """Trains the crystal transformer model. Can additionally evaluate on a test set.
    Supports updating existing models with new features like KV-caching.

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

    # Determine checkpoint path and update mode from Hydra config
    ckpt_path = config.get("ckpt_path")
    update_existing = config.get("update_existing", False)

    # Create or load model based on configuration
    if update_existing and ckpt_path and os.path.exists(ckpt_path):
        log.info(f"Loading existing model from checkpoint: {ckpt_path} for updating")
        # Load the model using the EnhancedCrystalTransformerModule
        model = EnhancedCrystalTransformerModule.load_from_checkpoint(ckpt_path)
        log.info("Successfully loaded model for updating with new features")
    else:
        # Create a new model instance using Hydra configuration
        log.info(f"Instantiating new model <{config.model._target_}>")
        model_instance = hydra.utils.instantiate(config.model)

        # Determine if we should use the enhanced module
        if config.get("use_enhanced_module", True):
            # Create enhanced module with the model's attributes
            model = EnhancedCrystalTransformerModule(
                **{
                    k: v
                    for k, v in vars(model_instance).items()
                    if not k.startswith("_") and k != "training"
                }
            )
            log.info("Created enhanced model with improved checkpoint handling")
        else:
            # Use the standard model
            model = model_instance

    # Set or update tokenizer config
    if tokenizer:
        model.tokenizer_config = {
            "idx_to_token": tokenizer.idx_to_token,
            "token_to_idx": tokenizer.vocab,
            "lattice_bins": tokenizer.lattice_bins,
            "coordinate_precision": tokenizer.coordinate_precision,
        }
        log.info("Set tokenizer configuration on model")

    # Configure ground truth hook for continuous regression losses
    if config.model.prediction_mode == "continuous":
        log.info("Setting up continuous regression loss hooks for continuous mode")

        # Set ground truth values from batch data during training
        def prepare_ground_truth_hook(pl_module, batch):
            # This hook extracts ground truth values from the batch data
            # and sets them in the model for regression loss calculation

            # Handle the case where pl_module might be a dict (happens in some distributed training scenarios)
            if isinstance(pl_module, dict):
                # This is normal in distributed training, so no need to log a warning
                return

            # Make sure pl_module has a model attribute
            if not hasattr(pl_module, "model"):
                log.warning("pl_module does not have a model attribute")
                return

            # Now we can safely access the model
            if hasattr(pl_module.model, "set_ground_truth_values"):
                # Extract lattice parameters and coordinates from the batch
                # This requires adapting your dataloader to provide this information
                if "ground_truth_lattice" in batch:
                    pl_module.model.set_ground_truth_values(
                        lattice_params=batch["ground_truth_lattice"],
                        fractional_coords=batch.get("ground_truth_coords"),
                    )

        # Register the hook with the model
        model.on_train_batch_start = prepare_ground_truth_hook

    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    # Set up minimal training if just updating the model
    if config.get("update_existing"):
        # Get update parameters from Hydra config
        update_epochs = config.get("update_epochs", 1)  # Default to 1 epoch for updates
        update_batches = config.get("update_batches", 10)  # Default to 10 batches

        # Set trainer parameters for minimal update training
        trainer.max_epochs = update_epochs
        trainer.limit_train_batches = update_batches

        log.info(
            f"Model update mode: Training for {update_epochs} epochs with {update_batches} batches"
        )

    # Don't load checkpoint again if we already loaded it for updating
    training_ckpt_path = (
        None if config.get("update_existing") else config.get("ckpt_path")
    )

    # Enable KV-caching for evaluation if specified in config
    if hasattr(model, "enable_kv_cache_for_evaluation"):
        use_kv_cache = config.get("use_kv_cache_for_evaluation", False)
        if use_kv_cache:
            log.info("Enabling KV-caching for model evaluation and generation")
            model.enable_kv_cache_for_evaluation(True)

    if config.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=data_module, ckpt_path=training_ckpt_path)

    if config.get("test"):
        log.info("Starting testing!")

        # Always use best checkpoint for testing unless specified otherwise
        test_ckpt_path = config.get("test_ckpt_path", "best")
        trainer.test(model=model, datamodule=data_module, ckpt_path=test_ckpt_path)

    try:
        return trainer.callback_metrics["val_loss"].item()
    except Exception:
        return None


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="train_hierarchical_crystal_transformer",
)
def main(config: DictConfig) -> Optional[float]:
    if config.get("debug"):
        log.info("Running in debug mode!")
        config.trainer.fast_dev_run = True

    # Example of how to use update_existing mode:
    # python train_crystal_transformer.py update_existing=true ckpt_path=/path/to/model.ckpt update_epochs=1 update_batches=5
    if config.get("update_existing"):
        log.info("Running in model update mode to incorporate code changes")
        if not config.get("ckpt_path"):
            log.error("ckpt_path must be provided when update_existing=true")
            return None

    return train(config)


if __name__ == "__main__":
    main()
