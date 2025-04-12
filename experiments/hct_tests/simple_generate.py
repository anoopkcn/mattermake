"""
Simple crystal structure generation script using a saved tokenizer config.
"""

import torch
import argparse
from pathlib import Path

from mattermake.models.hct_module import (
    HierarchicalCrystalTransformerModule,
)
from .save_load_tokenizer_config import load_tokenizer_config
from mattermake.utils.pylogger import get_pylogger

logger = get_pylogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate crystal structures using Mattermake with a saved tokenizer config"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--tokenizer-config",
        type=str,
        required=True,
        help="Path to saved tokenizer configuration JSON",
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
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (higher = more diverse)",
    )
    parser.add_argument(
        "--apply-wyckoff-constraints",
        action="store_true",
        help="Apply Wyckoff position constraints",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument(
        "--use-kv-cache",
        action="store_true",
        help="Enable KV-caching for faster generation",
    )

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    logger.info(f"Using device: {device}")

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Load model from checkpoint
    logger.info(f"Loading model from {args.checkpoint}...")
    model = HierarchicalCrystalTransformerModule.load_from_checkpoint(args.checkpoint)
    model.to(device)
    model.eval()

    # Load tokenizer config
    logger.info(f"Loading tokenizer config from {args.tokenizer_config}...")
    tokenizer_config = load_tokenizer_config(args.tokenizer_config)
    model.tokenizer_config = tokenizer_config

    # Force output to appear in stdout
    print(
        f"Tokenizer config loaded with {len(tokenizer_config.get('idx_to_token', {}))} tokens"
    )

    # Show some sample tokens for debugging
    idx_to_token = tokenizer_config.get("idx_to_token", {})
    sample_tokens = list(idx_to_token.items())[:5]
    print("Sample tokens:", sample_tokens)

    # Check if Wyckoff tokens are present
    wyckoff_tokens = [t for t in idx_to_token.values() if t.startswith("WYCK_")]
    print(f"Found {len(wyckoff_tokens)} Wyckoff position tokens: {wyckoff_tokens}")

    logger.info(
        f"Tokenizer config loaded with {len(tokenizer_config.get('idx_to_token', {}))} tokens"
    )

    # Set up constraints
    constraints = {
        "apply_wyckoff_constraints": args.apply_wyckoff_constraints,
        "token_id_maps": {
            "idx_to_token": tokenizer_config.get("idx_to_token", {}),
            "token_to_idx": tokenizer_config.get("token_to_idx", {}),
        },
    }

    logger.info(f"Generating {args.num_structures} crystal structures...")

    # Generate structures
    with torch.no_grad():
        structures = model.generate_structure(
            temperature=args.temperature,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.2,
            max_length=512,
            constraints=constraints,
            num_return_sequences=args.num_structures,
            verbose=True,
            use_kv_cache=args.use_kv_cache,
        )

    logger.info(f"Generated {len(structures)} structures.")

    # Process and save each structure
    for i, structure_data in enumerate(structures):
        # Extract hierarchical components
        composition = structure_data.get("composition", {})
        space_group = structure_data.get("space_group")
        lattice_params = structure_data.get("lattice_params", [])
        atoms = structure_data.get("atoms", [])

        # Print summary
        logger.info(f"\n--- Structure {i + 1} ---")
        logger.info(f"  Composition: {composition}")
        logger.info(f"  Space Group: {space_group}")
        if lattice_params and len(lattice_params) >= 6:
            logger.info(
                f"  Lattice: a={lattice_params[0]:.3f}, b={lattice_params[1]:.3f}, c={lattice_params[2]:.3f}"
            )
            logger.info(
                f"α={lattice_params[3]:.3f}, β={lattice_params[4]:.3f}, γ={lattice_params[5]:.3f}"
            )

        # Count Wyckoff tokens and show statistics
        wyckoff_positions = [
            atom.get("wyckoff") for atom in atoms if atom.get("wyckoff")
        ]
        unique_wyckoff = set(wyckoff_positions)
        logger.info(
            f"  Atoms: {len(atoms)}, Unique Wyckoff positions: {unique_wyckoff}"
        )

        # Save CIF file if possible
        try:
            if hasattr(structure_data, "to_pymatgen"):
                pmg_structure = structure_data.to_pymatgen()
                cif_path = output_path / f"structure_{i + 1}.cif"
                pmg_structure.to(filename=str(cif_path))
                logger.info(f"  Saved to: {cif_path}")
        except Exception as e:
            logger.error(f"  Error saving structure {i + 1}: {e}")

        # Save detailed structure data as JSON
        import json

        json_path = output_path / f"structure_{i + 1}.json"

        # Create serializable representation
        serializable_data = {
            "composition": composition,
            "space_group": space_group,
            "lattice_params": lattice_params,
            "atoms": [
                {
                    "element": atom.get("element"),
                    "wyckoff_position": atom.get("wyckoff"),
                    "coordinates": atom.get("coordinates", []).tolist()
                    if hasattr(atom.get("coordinates", []), "tolist")
                    else atom.get("coordinates", []),
                }
                for atom in atoms
            ],
        }

        with open(json_path, "w") as f:
            json.dump(serializable_data, f, indent=2)

    logger.info(f"Generation complete. Results saved to: {output_path}")


if __name__ == "__main__":
    main()

    # python -m mattermake.scripts.simple_generate \
    # --checkpoint path/to/your/model.ckpt \
    # --tokenizer-config tokenizer_config.json \
    # --output-dir generated_structures \
    # --num-structures 5 \
    # --apply-wyckoff-constraints \
    # --use-kv-cache \
    # --seed 42
