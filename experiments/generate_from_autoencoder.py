import os
import torch
import hydra
import rootutils
import json
from omegaconf import DictConfig

from src import utils
from src.utils.init import (
    configure_pytorch,
    patch_lightning_slurm_master_addr,
)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
log = utils.pylogger.get_pylogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="generate_autoencoder.yaml")
def generate(cfg: DictConfig):
    """Generate crystal structures using the trained autoencoder.

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

    # Load conditioning latent codes if specified
    latent_codes = None
    if cfg.latent_path:
        log.info(f"Loading latent codes from: {cfg.latent_path}")
        try:
            latent_codes = torch.load(cfg.latent_path, map_location=cfg.device)
        except:
            log.warning(f"Failed to load latent codes from {cfg.latent_path}")

    # Handle different generation modes
    if cfg.mode == "generate":
        # Generate new structures from random latents
        log.info(f"Generating {cfg.num_samples} new structures")

        structures = model.generate(
            batch_size=cfg.num_samples,
            num_atoms=cfg.num_atoms if cfg.num_atoms > 0 else None
        )

    elif cfg.mode == "reconstruct":
        # Load crystal structures to reconstruct
        if not cfg.input_path:
            raise ValueError("input_path must be provided for reconstruction mode")

        log.info(f"Loading input structures from: {cfg.input_path}")
        input_data = torch.load(cfg.input_path, map_location=cfg.device)

        # Reconstruct structures
        log.info("Reconstructing structures")
        structures = model.reconstruct(input_data)

    elif cfg.mode == "interpolate":
        # Load start and end structures
        if not cfg.start_path or not cfg.end_path:
            raise ValueError("start_path and end_path must be provided for interpolation mode")

        log.info(f"Loading start structure from: {cfg.start_path}")
        start_data = torch.load(cfg.start_path, map_location=cfg.device)

        log.info(f"Loading end structure from: {cfg.end_path}")
        end_data = torch.load(cfg.end_path, map_location=cfg.device)

        # Interpolate
        log.info(f"Generating {cfg.num_steps} interpolation steps")
        structures = model.interpolate(
            start_data,
            end_data,
            num_steps=cfg.num_steps
        )

    elif cfg.mode == "conditional":
        # Generate from provided latent codes
        if latent_codes is None:
            raise ValueError("latent_path must provide valid latent codes for conditional mode")

        log.info(f"Generating structures from {len(latent_codes)} provided latent codes")
        structures = model.decode(latent_codes)

    else:
        raise ValueError(f"Unknown generation mode: {cfg.mode}")

    # Convert tensors to serializable format
    serializable_structures = []

    # Handle different structure types based on mode
    if cfg.mode == "interpolate":
        # For interpolation, we have a list of structure dictionaries
        for i, struct_dict in enumerate(structures):
            entry = {}
            for key, tensor in struct_dict.items():
                entry[key] = tensor.cpu().numpy().tolist()
            entry["step"] = i
            serializable_structures.append(entry)
    else:
        # For other modes, we have a single dictionary with batch dimension
        for i in range(len(structures["atom_types"])):
            entry = {}
            for key in structures:
                entry[key] = structures[key][i].cpu().numpy().tolist()
            serializable_structures.append(entry)

    # Save results
    if cfg.output_path:
        os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)
        with open(cfg.output_path, "w") as f:
            json.dump(serializable_structures, f, indent=2)
        log.info(f"Saved {len(serializable_structures)} structures to {cfg.output_path}")

    log.info("Generation complete!")
    return serializable_structures


if __name__ == "__main__":
    patch_lightning_slurm_master_addr()
    configure_pytorch(log)
    generate()
