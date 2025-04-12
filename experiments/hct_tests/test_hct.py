import os
import torch
import argparse
import warnings
import lightning.pytorch as pl
from mattermake.data.hct_tokenizer import CrystalTokenData
from mattermake.data.hct_sequence_datamodule import CrystalSequenceDataModule
from mattermake.models.hct_module import (
    HierarchicalCrystalTransformerModule,
)
from mattermake.utils.pylogger import get_pylogger
import logging  # Import the logging library

# Suppress pymatgen warnings about fractional coordinates
warnings.filterwarnings(
    "ignore", message=".*fractional coordinates rounded to ideal values.*"
)

# Register CrystalTokenData for serialization
torch.serialization.add_safe_globals([CrystalTokenData])

# Configure the logger to output INFO messages to the console
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = get_pylogger(__name__)
# Ensure logger level is INFO (redundant if basicConfig is used, but safe)
logger.setLevel(logging.INFO)


def load_model_from_checkpoint(
    checkpoint_path,
    vocab_size=None,
    apply_wyckoff_constraints=False,
):
    """Load a model from checkpoint

    Args:
        checkpoint_path: Path to the checkpoint file
        vocab_size: Vocabulary size for the model
        apply_wyckoff_constraints: Whether to apply Wyckoff position constraints conditioned on space group
    """
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
        # Initialize the model with mixture density networks
        model = HierarchicalCrystalTransformerModule(
            vocab_size=vocab_size,
            coordinate_embedding_dim=4,  # Reduced from default (32) to avoid overflow
            lattice_mixture_components=5,  # Number of mixture components for lattice parameters
            coord_mixture_components=5    # Number of mixture components for coordinates
        )

        # Set Wyckoff constraints configuration after model initialization
        if hasattr(model, "model") and hasattr(model.model, "config"):
            logger.info(
                f"Setting apply_wyckoff_constraints to {apply_wyckoff_constraints}"
            )
            model.model.config.apply_wyckoff_constraints = apply_wyckoff_constraints
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


def generate_structures(
    model,
    num_structures=5,
    temperature=0.8,
    top_k=40,
    top_p=0.9,
    verbose=False,
):
    """Generate crystal structures using the trained model

    Args:
        model: The HierarchicalCrystalTransformerModule model
        num_structures: Number of structures to generate
        temperature: Sampling temperature
        top_k: If set, sample from top k most likely tokens
        top_p: If set, sample from tokens with cumulative probability >= top_p
        verbose: Whether to print detailed debugging information during generation

    Returns:
        List of generated structures
    """
    logger.info(f"Generating {num_structures} crystal structures")

    # Check model configuration
    if hasattr(model.model, "config"):
        logger.info(f"Active modules: {model.model.active_modules}")

        # Log Wyckoff constraints status
        if hasattr(model.model.config, "apply_wyckoff_constraints"):
            wyckoff_enabled = model.model.config.apply_wyckoff_constraints
            logger.info(
                f"Wyckoff position constraints: {'Enabled' if wyckoff_enabled else 'Disabled'}"
            )
    else:
        logger.warning("Could not access model config")

    logger.info("Using mixture density networks for predictions")
    if verbose:
        logger.info("Detailed generation debugging is enabled")

        # Check if the continuous prediction heads exist in the model
        if hasattr(model.model, "lattice_length_head"):
            logger.info("Model has lattice_length_head")
        if hasattr(model.model, "lattice_angle_head"):
            logger.info("Model has lattice_angle_head")
        if hasattr(model.model, "fractional_coord_head"):
            logger.info("Model has fractional_coord_head")

    # Use more controlled generation parameters to ensure complete structures
    structures = model.generate_structure(
        num_return_sequences=num_structures,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        verbose=verbose,
        # Set max_length longer to ensure we generate complete structures
        max_length=1024,
        # Use lower repetition penalty to allow repetition of element/coordinate tokens
        repetition_penalty=1.05,
    )

    # Check if mixture density predictions were properly generated
    has_continuous_lattice = any(
        s.get("used_continuous_lattice", False) for s in structures
    )
    has_continuous_coords = any(
        s.get("used_continuous_coords", False) for s in structures
    )
    logger.info(
        f"Generated structures with mixture density lattice: {has_continuous_lattice}"
    )
    logger.info(
        f"Generated structures with mixture density coordinates: {has_continuous_coords}"
    )

    return structures


def validate_structures(structures):
    """Basic validation of the generated structures

    Args:
        structures: List of generated crystal structures

    Returns:
        Dictionary with validation statistics
    """
    valid_count = 0
    issues = []

    # Track how many structures used mixture density predictions
    continuous_lattice_count = 0
    continuous_coords_count = 0

    # Track Wyckoff position statistics
    structures_with_wyckoff = 0
    wyckoff_positions_count = 0

    for i, structure in enumerate(structures):
        logger.info(f"\n--- Structure {i + 1} ---")

        # Log basic information
        logger.info(f"Composition: {structure['composition']}")
        logger.info(f"Space group: {structure['space_group']}")

        # Check if mixture density predictions were used
        if (
            "used_continuous_lattice" in structure
            and structure["used_continuous_lattice"]
        ):
            continuous_lattice_count += 1
            logger.info("Used mixture density lattice predictions: Yes")
        else:
            logger.info("Used mixture density lattice predictions: No")

        if (
            "used_continuous_coords" in structure
            and structure["used_continuous_coords"]
        ):
            continuous_coords_count += 1
            logger.info("Used mixture density coordinate predictions: Yes")
        else:
            logger.info("Used mixture density coordinate predictions: No")

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

        # Count Wyckoff positions usage
        has_wyckoff = any(
            "wyckoff" in atom and atom["wyckoff"] not in [None, "Unknown", ""]
            for atom in structure.get("atoms", [])
        )
        if has_wyckoff:
            structures_with_wyckoff += 1
            wyckoff_count = sum(
                1
                for atom in structure.get("atoms", [])
                if "wyckoff" in atom and atom["wyckoff"] not in [None, "Unknown", ""]
            )
            wyckoff_positions_count += wyckoff_count
            logger.info(
                f"Wyckoff positions: {wyckoff_count}/{len(structure.get('atoms', []))} atoms"
            )
        else:
            logger.info("Wyckoff positions: None")

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
    logger.info(
        f"Structures using mixture density lattice predictions: {continuous_lattice_count}/{len(structures)}"
    )
    logger.info(
        f"Structures using mixture density coordinate predictions: {continuous_coords_count}/{len(structures)}"
    )

    if issues:
        logger.info("Issues found:")
        for issue in issues:
            logger.info(f"- {issue}")

    return {
        "valid_count": valid_count,
        "mixture_density_lattice_count": continuous_lattice_count,
        "mixture_density_coords_count": continuous_coords_count,
        "structures_with_wyckoff": structures_with_wyckoff,
        "wyckoff_positions_count": wyckoff_positions_count,
    }


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
    # Note: Removed --continuous flag as mixture density networks are always used
    parser.add_argument(
        "--limit_structures",
        type=int,
        default=None,
        help="Limit number of structures to load for testing (e.g. 1000)",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed debugging output during generation",
    )
    parser.add_argument(
        "--apply_wyckoff_constraints",
        action="store_true",
        help="Apply Wyckoff position constraints conditioned on space group",
    )
    args = parser.parse_args()

    # Set up data module
    logger.info(f"Loading data from {args.data_dir}")

    # Setup filter function to limit number of structures if requested
    train_filter = None
    if args.limit_structures is not None:
        logger.info(f"Limiting dataset to {args.limit_structures} structures")

        # Counter for limiting structures
        counter = [0]  # Using list to allow modification within closure
        max_structures = args.limit_structures

        def limit_structures_filter(item):
            # Only accept structures until we reach the limit
            counter[0] += 1
            return counter[0] <= max_structures

        train_filter = limit_structures_filter

    data_module = CrystalSequenceDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        train_filter=train_filter,
        val_filter=train_filter,  # Use same filter for validation
        cache_tokenized=False,
    )

    # Prepare data module
    data_module.prepare_data()
    data_module.setup(stage="fit")

    # Get tokenizer for vocabulary size
    tokenizer = data_module.get_tokenizer()
    vocab_size = tokenizer.vocab_size if tokenizer else 2000  # Default if not available

    # Load or initialize model
    model = load_model_from_checkpoint(
        args.checkpoint,
        vocab_size=vocab_size,
        apply_wyckoff_constraints=args.apply_wyckoff_constraints,
    )

    # Force configuration to match the arguments
    # This is needed because loaded checkpoint might have been trained with different settings
    if hasattr(model.model, "config"):

        # Set Wyckoff constraints configuration
        logger.info(
            f"Setting Wyckoff position constraints: {'Enabled' if args.apply_wyckoff_constraints else 'Disabled'}"
        )
        model.model.config.apply_wyckoff_constraints = args.apply_wyckoff_constraints

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
        model,
        num_structures=args.num_structures,
        temperature=args.temperature,
        top_k=40,  # Default value
        top_p=0.9,  # Default value
        verbose=args.verbose,  # Use verbose flag for detailed debugging
    )

    # Validate structures
    validation_results = validate_structures(structures)

    # Log validation summary
    logger.info("\nValidation Summary:")
    logger.info(f"Valid structures: {validation_results['valid_count']}")
    logger.info(
        f"With mixture density lattice: {validation_results['mixture_density_lattice_count']}"
    )
    logger.info(
        f"With mixture density coordinates: {validation_results['mixture_density_coords_count']}"
    )
    logger.info(
        f"Structures with Wyckoff positions: {validation_results['structures_with_wyckoff']}/{args.num_structures}"
    )
    logger.info(
        f"Total Wyckoff positions assigned: {validation_results['wyckoff_positions_count']}"
    )


if __name__ == "__main__":
    main()

    # Usage:
    # python -m mattermake.scripts.test_crystal_transformer_fixed \
    #     --data_dir data/structure_tokens \
    #     --train
    #     --limit_structures 1000 \
    #     --num_structures 5 \
    #     --temperature 0.8 \
    #     --batch_size 16 \
    #     --gpu
