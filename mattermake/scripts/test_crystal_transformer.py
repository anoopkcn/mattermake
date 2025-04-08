import os
import torch
import argparse
import warnings
import lightning.pytorch as pl
from mattermake.data.crystal_tokenizer import CrystalTokenData
from mattermake.data.crystal_sequence_datamodule import CrystalSequenceDataModule
from mattermake.models.hierarchical_crystal_transformer_module import (
    HierarchicalCrystalTransformerModule,
)
from mattermake.utils.pylogger import get_pylogger

# Suppress pymatgen warnings about fractional coordinates
warnings.filterwarnings("ignore", message=".*fractional coordinates rounded to ideal values.*")

# Uncomment if needed
# from torch.utils.data import DataLoader

# Register CrystalTokenData for serialization
torch.serialization.add_safe_globals([CrystalTokenData])

logger = get_pylogger(__name__)


def load_model_from_checkpoint(checkpoint_path, vocab_size=None):
    """Load a model from checkpoint"""
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        logger.info(f"Loading model from checkpoint: {checkpoint_path}")
        model = HierarchicalCrystalTransformerModule.load_from_checkpoint(
            checkpoint_path
        )
        return model
    else:
        if checkpoint_path is None:
            logger.info("No checkpoint path provided")
        else:
            logger.info(f"Checkpoint not found: {checkpoint_path}")
        logger.info("Initializing new model")
        if vocab_size is None:
            raise ValueError(
                "vocab_size must be provided when initializing a new model"
            )
        model = HierarchicalCrystalTransformerModule(vocab_size=vocab_size)
        return model


def train_model(model, data_module, max_epochs=5, gpus=1):
    """Train the model for a few epochs"""
    logger.info(f"Training model for {max_epochs} epochs")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if gpus > 0 and torch.cuda.is_available() else "cpu",
        devices=gpus if gpus > 0 and torch.cuda.is_available() else None,
        precision="16-mixed" if torch.cuda.is_available() else "32",
        log_every_n_steps=10,
    )

    trainer.fit(model, data_module)
    return model


def generate_structures(model, num_structures=5, temperature=0.8, top_k=40, top_p=0.9):
    """Generate crystal structures using the trained model"""
    logger.info(f"Generating {num_structures} crystal structures")

    structures = model.generate_structure(
        num_return_sequences=num_structures,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    return structures


def validate_structures(structures):
    """Basic validation of the generated structures"""
    valid_count = 0
    issues = []

    for i, structure in enumerate(structures):
        logger.info(f"\n--- Structure {i + 1} ---")

        # Log basic information
        logger.info(f"Composition: {structure['composition']}")
        logger.info(f"Space group: {structure['space_group']}")

        # Check if lattice parameters are present and valid
        if "lattice_params" in structure and len(structure["lattice_params"]) >= 6:
            a, b, c, alpha, beta, gamma = structure["lattice_params"][:6]
            logger.info(
                f"Lattice: a={a:.3f}, b={b:.3f}, c={c:.3f}, α={alpha:.3f}°, β={beta:.3f}°, γ={gamma:.3f}°"
            )

            # Basic validation
            if min(a, b, c) <= 0:
                issues.append(f"Structure {i + 1}: Invalid lattice lengths")
            if min(alpha, beta, gamma) <= 0 or max(alpha, beta, gamma) >= 180:
                issues.append(f"Structure {i + 1}: Invalid lattice angles")
        else:
            issues.append(
                f"Structure {i + 1}: Missing or incomplete lattice parameters"
            )

        # Check atoms
        if "atoms" in structure and structure["atoms"]:
            atom_count = len(structure["atoms"])
            logger.info(f"Atoms: {atom_count}")

            # Print first few atoms
            for j, atom in enumerate(structure["atoms"][: min(3, atom_count)]):
                coords_str = (
                    ", ".join(f"{x:.3f}" for x in atom["coords"])
                    if "coords" in atom
                    else "N/A"
                )
                logger.info(
                    f"  Atom {j + 1}: {atom.get('element', 'Unknown')} at [{coords_str}] (Wyckoff: {atom.get('wyckoff', 'Unknown')})"
                )

            # Basic validation
            valid_atoms = all(
                "element" in atom and "coords" in atom and len(atom["coords"]) >= 3
                for atom in structure["atoms"]
            )
            if not valid_atoms:
                issues.append(f"Structure {i + 1}: Invalid atom specifications")
        else:
            issues.append(f"Structure {i + 1}: No atoms specified")

        # Check if structure can be converted to pymatgen Structure
        if "pmg_structure" in structure:
            logger.info("Successfully converted to pymatgen Structure")
            valid_count += 1
        elif "structure_error" in structure:
            logger.info(
                f"Failed to convert to pymatgen Structure: {structure['structure_error']}"
            )
            issues.append(f"Structure {i + 1}: {structure['structure_error']}")

    logger.info(f"\nValid structures: {valid_count}/{len(structures)}")
    if issues:
        logger.info("Issues found:")
        for issue in issues:
            logger.info(f"- {issue}")

    return valid_count


def main():
    parser = argparse.ArgumentParser(description="Test Crystal Transformer Model")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory with processed data"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--train", action="store_true", help="Train the model if specified"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs to train"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--num_structures", type=int, default=5, help="Number of structures to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature"
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    args = parser.parse_args()

    # Set up data module
    logger.info(f"Loading data from {args.data_dir}")
    data_module = CrystalSequenceDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
    )

    # Prepare data module
    data_module.prepare_data()
    data_module.setup(stage="fit")

    # Get tokenizer for vocabulary size
    tokenizer = data_module.get_tokenizer()
    vocab_size = tokenizer.vocab_size if tokenizer else 2000  # Default if not available

    # Load or initialize model
    model = load_model_from_checkpoint(args.checkpoint, vocab_size=vocab_size)

    # Set tokenizer config in the model for proper decoding
    if tokenizer:
        model.tokenizer_config = {
            "idx_to_token": tokenizer.idx_to_token,
            "token_to_idx": tokenizer.vocab,
            "lattice_bins": tokenizer.lattice_bins,
            "coordinate_precision": tokenizer.coordinate_precision,
        }

    # Train if requested
    if args.train:
        model = train_model(
            model, data_module, max_epochs=args.epochs, gpus=1 if args.gpu else 0
        )

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    model = model.to(device)

    # Generate structures
    structures = generate_structures(
        model, num_structures=args.num_structures, temperature=args.temperature
    )

    # Validate structures
    validate_structures(structures)


if __name__ == "__main__":
    main()

    # Usage:
    # python -m mattermake.scripts.test_crystal_transformer \
    #     --data_dir processed_data \
    #     --train \
    #     --epochs 5 \
    #     --batch_size 16 \
    #     --num_structures 5 \
    #     --temperature 0.8 \
    #     --gpu
