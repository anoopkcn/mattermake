"""
Script for generating slices from trained models.
"""
import torch
import hydra
import rootutils
import json
import os

import numpy as np
from omegaconf import DictConfig
from lightning.pytorch import LightningModule
import lightning.pytorch as pl

from src.utils.vocab import decode_slice

from src import utils
from src.utils.init import (
    configure_pytorch,
    # init_distributed_mode,
    log_distributed_settings,
    patch_lightning_slurm_master_addr,
)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = utils.pylogger.get_pylogger(__name__)

@hydra.main(version_base="1.3", config_path="../configs", config_name="generate.yaml")
def generate(cfg: DictConfig):
    """Generate slices from a trained model using embeddings.

    Args:
        cfg: Configuration composed by Hydra.
    """
    # Load model from checkpoint
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    if cfg.ckpt_path:
        log.info(f"Loading weights from checkpoint: {cfg.ckpt_path}")
        checkpoint = torch.load(cfg.ckpt_path, map_location=cfg.device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])

    model = model.to(cfg.device)
    model.eval()

    # Load data for generation
    log.info(f"Loading data from: {cfg.data_path}")
    try:
        # Try loading with torch.load since file is .pt format
        data = torch.load(cfg.data_path)
    except Exception:
        # Fall back to numpy if needed
        data = np.load(cfg.data_path, allow_pickle=True).item()

    embeddings = data["embeddings"][:cfg.num_samples]

    if "slice_ids" in data and cfg.show_original:
        true_slices = [decode_slice(s) for s in data["slice_ids"][:cfg.num_samples]]
    else:
        true_slices = None

    log.info("Generating slices...")
    results = []

    for i, embedding in enumerate(embeddings):
        log.info(f"Generating sample {i+1}/{cfg.num_samples}")
        embedding_tensor = torch.tensor(embedding).float().unsqueeze(0).to(cfg.device) if not torch.is_tensor(embedding) else embedding.float().unsqueeze(0).to(cfg.device)

        # Generate using the model's generate method without passing the idx parameter
        generated_slices = model.generate(
            embeddings=embedding_tensor,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_k=cfg.top_k
        )

        # Get the first generated slice
        decoded_slice = generated_slices[0]
        log.info(f"Generated: {decoded_slice}")

        if true_slices and i < len(true_slices):
            log.info(f"Original:  {true_slices[i]}")

        log.info("---")

        results.append({
            "generated": decoded_slice,
            "original": true_slices[i] if true_slices and i < len(true_slices) else None
        })

    # Save results if specified
    if cfg.output_path:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)

        with open(cfg.output_path, 'w') as f:
            json.dump(results, f, indent=2)
        log.info(f"Saved results to {cfg.output_path}")

    log.info("Generation complete!")

    return results

if __name__ == "__main__":
    pl.seed_everything(42)
    patch_lightning_slurm_master_addr()
    # init_distributed_mode(port=12354)
    log_distributed_settings(log)
    configure_pytorch(log)
    generate()
