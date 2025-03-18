import os
import torch
import hydra
import rootutils
import json
from omegaconf import DictConfig
from tqdm import tqdm

from src import utils
from src.utils.init import (
    configure_pytorch,
    patch_lightning_slurm_master_addr,
)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
log = utils.pylogger.get_pylogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="generate_crystals.yaml")
def generate(cfg: DictConfig):
    """Generate crystal structures using a trained diffusion model.

    Args:
        cfg: Configuration composed by Hydra.
    """
    # Load model from checkpoint
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)

    if cfg.ckpt_path:
        log.info(f"Loading weights from checkpoint: {cfg.ckpt_path}")
        checkpoint = torch.load(cfg.ckpt_path, map_location=cfg.device)
        model.load_state_dict(checkpoint["state_dict"])

    # Move model to specified device
    model = model.to(cfg.device)
    model.eval()

    # Load conditioning data for property-conditioned generation if specified
    conditions = None
    if cfg.condition_path:
        log.info(f"Loading conditioning data from: {cfg.condition_path}")
        try:
            conditions = torch.load(cfg.condition_path, map_location=cfg.device)
        except:
            log.warning(f"Failed to load conditioning data from {cfg.condition_path}")

    # Set up batch generation
    total_structures = cfg.num_samples
    batch_size = cfg.generation_batch_size
    num_batches = (total_structures + batch_size - 1) // batch_size

    log.info(f"Generating {total_structures} structures with batch size {batch_size}...")

    generated_structures = []

    for i in tqdm(range(num_batches)):
        current_batch_size = min(batch_size, total_structures - i * batch_size)

        # Get conditions for this batch if available
        batch_conditions = None
        if conditions is not None and i * batch_size < len(conditions):
            batch_end = min((i + 1) * batch_size, len(conditions))
            batch_conditions = conditions[i * batch_size:batch_end]

            # Handle shape mismatch if needed
            if batch_conditions.shape[0] != current_batch_size:
                batch_conditions = batch_conditions[:current_batch_size]

        # Generate batch of structures
        with torch.no_grad():
            batch_structures = model.generate(
                batch_size=current_batch_size,
                num_atoms=cfg.num_atoms if cfg.num_atoms > 0 else None,
                condition=batch_conditions
            )

        # Convert tensors to numpy/lists for JSON serialization
        serializable_batch = []
        for j in range(current_batch_size):
            structure = {
                "atom_types": batch_structures["atom_types"][j].cpu().numpy().tolist(),
                "positions": batch_structures["positions"][j].cpu().numpy().tolist(),
                "lattice": batch_structures["lattice"][j].cpu().numpy().tolist()
            }
            serializable_batch.append(structure)

        generated_structures.extend(serializable_batch)

    # Save results
    if cfg.output_path:
        os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)

        with open(cfg.output_path, "w") as f:
            json.dump(generated_structures, f, indent=2)

        log.info(f"Saved {len(generated_structures)} structures to {cfg.output_path}")

    log.info("Generation complete!")
    return generated_structures


if __name__ == "__main__":
    patch_lightning_slurm_master_addr()
    configure_pytorch(log)
    generate()
