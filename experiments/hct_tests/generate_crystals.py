"""
Generation script for Mattermake model.
Generates crystal structures using a pretrained hierarchical crystal transformer.
"""

import os
import torch
import argparse
import warnings
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
from typing import Optional


from mattermake.data.hct_tokenizer import CrystalTokenData
from mattermake.data.hct_sequence_datamodule import CrystalSequenceDataModule
from mattermake.models.hct_module import (
    HierarchicalCrystalTransformerModule,
)
from mattermake.utils.pylogger import get_pylogger

# Suppress pymatgen warnings about fractional coordinates
warnings.filterwarnings(
    "ignore", message=".*fractional coordinates rounded to ideal values.*"
)

# Register CrystalTokenData for serialization
torch.serialization.add_safe_globals([CrystalTokenData])

logger = get_pylogger(__name__)


def load_model_from_checkpoint(checkpoint_path):
    """Load a model from checkpoint"""
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        model = HierarchicalCrystalTransformerModule.load_from_checkpoint(
            checkpoint_path
        )
        return model
    else:
        if checkpoint_path is None:
            raise ValueError("No checkpoint path provided")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


def generate_crystals(
    checkpoint_path: str,
    output_dir: str = "generated_structures",
    data_dir: Optional[str] = None,
    num_structures: int = 10,
    seed: Optional[int] = None,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    max_length: int = 512,
    apply_wyckoff_constraints: bool = True,
    gpu: bool = True,
    verbose: bool = False,
    save_json: bool = True,
):
    """
    Generate crystal structures using a pretrained hierarchical crystal transformer.

    Args:
        checkpoint_path: Path to the model checkpoint
        output_dir: Directory to save generated structures
        data_dir: Directory with processed data (for tokenizer)
        num_structures: Number of structures to generate
        seed: Random seed for reproducibility
        temperature: Sampling temperature (higher = more diverse)
        top_k: If set, sample from top k most likely tokens
        top_p: If set, sample from tokens with cumulative probability >= top_p
        repetition_penalty: Penalty for repeating tokens
        max_length: Maximum sequence length
        apply_wyckoff_constraints: Whether to apply Wyckoff position constraints
        gpu: Whether to use GPU if available
        verbose: Whether to print verbose output during generation
        save_json: Whether to save structure data as JSON
    """
    # Set random seed for reproducibility if provided
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    logger.info(f"Loading model from {checkpoint_path}...")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    logger.info(f"Using device: {device}")

    # Load the model from checkpoint
    model = load_model_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()

    # Load tokenizer if data_dir is provided
    if data_dir:
        logger.info(f"Loading tokenizer from {data_dir}")
        data_module = CrystalSequenceDataModule(data_dir=data_dir, batch_size=1)
        data_module.prepare_data()
        data_module.setup(stage="fit")

        tokenizer = data_module.get_tokenizer()
        if tokenizer:
            logger.info("Tokenizer loaded successfully")
            model.tokenizer_config = {
                "idx_to_token": tokenizer.idx_to_token,
                "token_to_idx": tokenizer.vocab,
                "lattice_bins": getattr(tokenizer, "lattice_bins", None),
                "coordinate_precision": getattr(
                    tokenizer, "coordinate_precision", None
                ),
            }
        else:
            logger.warning("No tokenizer found in data_module")

    logger.info(f"Generating {num_structures} crystal structures...")

    # Prepare constraints
    constraints = {"apply_wyckoff_constraints": apply_wyckoff_constraints}

    # Generate structures
    generated_structures = []

    with torch.no_grad():
        structures = model.generate_structure(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_length=max_length,
            constraints=constraints,
            num_return_sequences=num_structures,
            verbose=verbose,
        )

        generated_structures.extend(structures)

    logger.info(f"Generated {len(generated_structures)} structures.")
    logger.info("Converting to Pymatgen structures and saving...")

    # Process and save each structure
    structure_summaries = []

    for i, structure_data in enumerate(tqdm(generated_structures)):
        # Extract hierarchical components
        composition = structure_data.get("composition", {})
        space_group = structure_data.get("space_group")
        lattice_params = structure_data.get("lattice_params", [])
        atoms = structure_data.get("atoms", [])

        # Create a summary for this structure
        summary = {
            "id": i + 1,
            "composition": composition,
            "space_group": space_group,
            "lattice_params": lattice_params,
            "num_atoms": len(atoms),
        }
        structure_summaries.append(summary)

        # Print summary
        logger.info(f"\n--- Structure {i + 1} ---")
        logger.info(f"  Composition: {composition}")
        logger.info(f"  Space Group: {space_group}")
        if lattice_params and len(lattice_params) >= 6:
            logger.info(
                f"  Lattice: a={lattice_params[0]:.3f}, b={lattice_params[1]:.3f}, c={lattice_params[2]:.3f}"
            )
            logger.info(
                f"           α={lattice_params[3]:.3f}, β={lattice_params[4]:.3f}, γ={lattice_params[5]:.3f}"
            )
        logger.info(f"  Atoms: {len(atoms)}")

        if atoms and len(atoms) > 0:
            # Show a few atoms as example
            for j, atom in enumerate(atoms[: min(3, len(atoms))]):
                element = atom.get("element", "Unknown")
                wyckoff = atom.get("wyckoff_position", "Unknown")
                coords = atom.get("coordinates", [])
                if coords:
                    coords_str = ", ".join([f"{c:.3f}" for c in coords])
                    logger.info(
                        f"    Atom {j + 1}: {element} at [{coords_str}] (Wyckoff: {wyckoff})"
                    )

        # Save CIF file
        try:
            # Try to convert to pymatgen structure and save as CIF
            if hasattr(structure_data, "to_pymatgen"):
                pmg_structure = structure_data.to_pymatgen()
                cif_path = output_path / f"structure_{i + 1}.cif"
                pmg_structure.to(filename=str(cif_path))
                logger.info(f"  Saved to: {cif_path}")

                # Add structure info
                summary["formula"] = pmg_structure.composition.reduced_formula
                summary["volume"] = float(pmg_structure.volume)
            else:
                logger.warning(
                    f"  Structure {i + 1} could not be converted to Pymatgen format"
                )
        except Exception as e:
            logger.error(f"  Error saving structure {i + 1}: {e}")

        # Save detailed structure data as JSON
        if save_json:
            json_path = output_path / f"structure_{i + 1}.json"

            # Create serializable representation
            serializable_data = {
                "composition": composition,
                "space_group": space_group,
                "lattice_params": lattice_params,
                "atoms": [
                    {
                        "element": atom.get("element"),
                        "wyckoff_position": atom.get("wyckoff_position"),
                        "coordinates": atom.get("coordinates", []).tolist()
                        if hasattr(atom.get("coordinates", []), "tolist")
                        else atom.get("coordinates", []),
                    }
                    for atom in atoms
                ],
            }

            with open(json_path, "w") as f:
                json.dump(serializable_data, f, indent=2)

    # Save summary of all structures
    summary_path = output_path / "generation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "generation_parameters": {
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty,
                    "max_length": max_length,
                    "seed": seed,
                    "apply_wyckoff_constraints": apply_wyckoff_constraints,
                },
                "structures": structure_summaries,
            },
            f,
            indent=2,
        )

    logger.info(
        f"\nGeneration complete. Generated {len(generated_structures)} structures."
    )
    logger.info(f"Results saved to: {output_path}")

    return generated_structures


def main():
    parser = argparse.ArgumentParser(
        description="Generate crystal structures using Mattermake"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory with processed data for tokenizer",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="generated_structures",
        help="Directory to save generated structures",
    )
    parser.add_argument(
        "--num-structures",
        type=int,
        default=3,
        help="Number of structures to generate",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (higher = more diverse)",
    )
    parser.add_argument(
        "--top-k", type=int, default=40, help="Sample from top k most likely tokens"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Sample from tokens with cumulative probability >= top_p",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.2,
        help="Penalty for repeating tokens",
    )
    parser.add_argument(
        "--max-length", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument(
        "--apply-wyckoff-constraints",
        action="store_true",
        help="Apply Wyckoff position constraints",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument(
        "--verbose", action="store_true", help="Print verbose output during generation"
    )

    args = parser.parse_args()

    generate_crystals(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        num_structures=args.num_structures,
        seed=args.seed,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_length=args.max_length,
        apply_wyckoff_constraints=args.apply_wyckoff_constraints,
        gpu=args.gpu,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
