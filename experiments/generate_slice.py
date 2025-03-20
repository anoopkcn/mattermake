"""
Script for generating slices from trained models.
"""

import torch
import hydra
import rootutils
import json
import os
import tqdm

import numpy as np
from omegaconf import DictConfig
from lightning.pytorch import LightningModule
import lightning.pytorch as pl

from src.utils.vocab import decode_slice

from src import utils
from src.utils.init import (
    configure_pytorch,
    init_distributed_mode,
    log_distributed_settings,
    patch_lightning_slurm_master_addr,
)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = utils.pylogger.get_pylogger(__name__)


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="generate_slice.yaml"
)
def generate(cfg: DictConfig):
    """Generate slices from a trained model using embeddings.

    Args:
        cfg: Configuration composed by Hydra.
    """
    # Load model from checkpoint
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    if cfg.ckpt_path:
        log.info(f"Loading weights from checkpoint on CPU: {cfg.ckpt_path}")
        checkpoint = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])

        # Now move to GPU and convert to half precision
        model = model.to(cfg.device)
        if cfg.get("use_half_precision", True):
            model = model.half()

    # Apply optimizations
    model = model.to(cfg.device)
    if cfg.get("use_half_precision", True):
        log.info("Using half precision (float16) for inference")
        model = model.half()

    if cfg.get("use_compile", False) and hasattr(torch, "compile"):
        log.info("Using torch.compile to optimize model for inference")
        model.model = torch.compile(model.model, mode="reduce-overhead")

    model.eval()

    # Load data for generation
    log.info(f"Loading data from: {cfg.data_path}")
    try:
        # Try loading with torch.load since file is .pt format
        data = torch.load(cfg.data_path)
    except Exception:
        # Fall back to numpy if needed
        data = np.load(cfg.data_path, allow_pickle=True).item()

    # Use all validation samples or limit by cfg.num_samples
    num_samples = (
        len(data["embeddings"])
        if cfg.get("evaluate_all", False)
        else min(cfg.num_samples, len(data["embeddings"]))
    )
    total_embeddings = data["embeddings"][:num_samples]

    if "slice_ids" in data and cfg.show_original:
        true_slices = [decode_slice(s) for s in data["slice_ids"][:num_samples]]
    else:
        true_slices = None
        log.warning(
            "No original slices found in data. Only generated slices will be saved."
        )

    # Configure batch size for generation
    batch_size = cfg.get("generation_batch_size", 32)
    log.info(f"Generating {num_samples} slices with batch size {batch_size}...")

    results = []

    # Process in batches
    for i in tqdm.tqdm(range(0, num_samples, batch_size), desc="Generating batches"):
        batch_embeddings = total_embeddings[i : min(i + batch_size, num_samples)]

        # Handle different embedding formats correctly
        if isinstance(batch_embeddings, torch.Tensor):
            # If already a tensor (which might be the case with data["embeddings"])
            embedding_tensor = batch_embeddings.to(cfg.device)
        elif isinstance(batch_embeddings, list) and torch.is_tensor(
            batch_embeddings[0]
        ):
            # If list of tensors
            embedding_tensor = torch.stack(batch_embeddings).to(cfg.device)
        else:
            # If numpy arrays or other format
            embedding_tensor = torch.tensor(batch_embeddings, dtype=torch.float).to(
                cfg.device
            )

        if cfg.get("use_half_precision", True):
            embedding_tensor = embedding_tensor.half()

        # Generate slices for the entire batch
        with torch.no_grad():
            if hasattr(model, "batch_generate"):
                # Use batch_generate if available
                batch_outputs = model.batch_generate(
                    embeddings=embedding_tensor,
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.temperature,
                    top_k=cfg.top_k,
                )
            else:
                # Fall back to individual generation
                batch_outputs = []
                for j in range(embedding_tensor.size(0)):
                    single_embedding = embedding_tensor[
                        j : j + 1
                    ]  # Keep batch dimension
                    generated = model.generate(
                        embeddings=single_embedding,
                        max_new_tokens=cfg.max_new_tokens,
                        temperature=cfg.temperature,
                        top_k=cfg.top_k,
                    )
                    batch_outputs.append(
                        generated[0]
                    )  # First element from each generation

        # Process results
        for j, decoded_slice in enumerate(batch_outputs):
            global_idx = i + j

            # Only print details for a few samples when evaluating all
            if cfg.get("evaluate_all", False) and global_idx < 5:
                log.info(f"Sample {global_idx + 1} Generated: {decoded_slice}")
                if true_slices and global_idx < len(true_slices):
                    log.info(
                        f"Sample {global_idx + 1} Original:  {true_slices[global_idx]}"
                    )
                log.info("---")
            elif not cfg.get("evaluate_all", False):
                log.info(f"Generated: {decoded_slice}")
                if true_slices and global_idx < len(true_slices):
                    log.info(f"Original:  {true_slices[global_idx]}")
                log.info("---")

            results.append(
                {
                    "generated": decoded_slice,
                    "original": true_slices[global_idx]
                    if true_slices and global_idx < len(true_slices)
                    else None,
                }
            )

    # Save results if specified
    if cfg.output_path:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)

        with open(cfg.output_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"Saved {len(results)} results to {cfg.output_path}")

    log.info("Generation complete!")

    return results


if __name__ == "__main__":
    pl.seed_everything(42)
    patch_lightning_slurm_master_addr()
    init_distributed_mode(port=12354)
    log_distributed_settings(log)
    configure_pytorch(log)
    generate()
