import torch
import hydra
import rootutils
import json
import os
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig

from src import utils
from src.utils.init import configure_pytorch, patch_lightning_slurm_master_addr

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
log = utils.pylogger.get_pylogger(__name__)


def compute_metrics(original, reconstructed):
    """Compute metrics between original and reconstructed structures"""
    metrics = {}

    # Atom type accuracy
    atom_mask = original.get('atom_mask', torch.ones_like(original['atom_types']).bool())
    orig_types = original['atom_types'][atom_mask]
    recon_types = reconstructed['atom_types'][atom_mask]
    metrics['atom_type_accuracy'] = (orig_types == recon_types).float().mean().item()

    # Position RMSE
    orig_pos = original['positions'][atom_mask]
    recon_pos = reconstructed['positions'][atom_mask]

    # Handle periodic boundary conditions by finding minimum distance
    pos_diff = torch.abs(orig_pos - recon_pos)
    pos_diff = torch.min(pos_diff, 1.0 - pos_diff)  # Account for periodicity
    metrics['position_rmse'] = torch.sqrt(torch.mean(torch.sum(pos_diff**2, dim=-1))).item()

    # Lattice parameter RMSE
    metrics['lattice_rmse'] = torch.sqrt(torch.mean((original['lattice'] - reconstructed['lattice'])**2)).item()

    return metrics

@hydra.main(version_base="1.3", config_path="../configs", config_name="validate_autoencoder.yaml")
def validate(cfg: DictConfig):
    """Validate the crystal autoencoder by testing reconstruction quality.

    Args:
        cfg: Configuration composed by Hydra.
    """
    # Load model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)

    if cfg.ckpt_path:
        log.info(f"Loading weights from checkpoint: {cfg.ckpt_path}")
        checkpoint = torch.load(cfg.ckpt_path, map_location=cfg.device)
        model.load_state_dict(checkpoint["state_dict"])

    model = model.to(cfg.device)
    model.eval()

    # Load validation data
    log.info(f"Loading validation data from: {cfg.val_data_path}")
    val_data = torch.load(cfg.val_data_path, map_location=cfg.device)

    dataloader = None
    if isinstance(val_data, dict) and "crystal" in val_data:
        # Data is already in batch format
        batch = val_data
        sample_indices = list(range(min(cfg.num_samples, len(batch["crystal"]["atom_types"]))))
    else:
        # Load from data module
        log.info(f"Instantiating data module <{cfg.data._target_}>")
        datamodule = hydra.utils.instantiate(cfg.data)
        datamodule.setup(stage="validate")
        dataloader = datamodule.val_dataloader()

        # Get first batch
        batch = next(iter(dataloader))
        sample_indices = list(range(min(cfg.num_samples, len(batch["crystal"]["atom_types"]))))

    # Run reconstruction
    log.info("Running reconstructions...")
    results = []

    with torch.no_grad():
        for idx in tqdm(sample_indices, desc="Reconstructing samples"):
            # Extract single sample
            sample = {
                "crystal": {
                    key: batch["crystal"][key][idx:idx+1] for key in batch["crystal"]
                }
            }
            if "condition" in batch:
                sample["condition"] = batch["condition"][idx:idx+1]

            # Move to device
            for key in sample["crystal"]:
                sample["crystal"][key] = sample["crystal"][key].to(cfg.device)
            if "condition" in sample:
                sample["condition"] = sample["condition"].to(cfg.device)

            # Encode
            latent = model.encode(sample["crystal"])

            # Decode
            reconstructed = model.decode(latent)

            # Compute metrics
            metrics = compute_metrics(sample["crystal"], reconstructed)

            # Add to results
            result = {
                "sample_idx": idx,
                "metrics": metrics,
                "original": {k: v[0].cpu().numpy().tolist() for k, v in sample["crystal"].items() if k != "atom_mask"},
                "reconstructed": {k: v[0].cpu().numpy().tolist() for k, v in reconstructed.items()}
            }
            results.append(result)

            # Print metrics for first few samples
            if idx < 5:
                log.info(f"Sample {idx} metrics: {metrics}")

    # Compute average metrics
    avg_metrics = {metric: np.mean([r["metrics"][metric] for r in results]) for metric in results[0]["metrics"]}
    log.info(f"Average metrics over {len(results)} samples:")
    for metric, value in avg_metrics.items():
        log.info(f"  {metric}: {value:.4f}")

    # Save results
    if cfg.output_path:
        output_data = {
            "avg_metrics": avg_metrics,
            "samples": results
        }

        os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)
        with open(cfg.output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        log.info(f"Saved validation results to {cfg.output_path}")

    return avg_metrics


if __name__ == "__main__":
    patch_lightning_slurm_master_addr()
    configure_pytorch(log)
    validate()
