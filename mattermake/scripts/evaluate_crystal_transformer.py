import os
import json
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

# from pathlib import Path
# from pymatgen.core import Structure
from pymatgen.analysis.structure_analyzer import VoronoiConnectivity
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.dimensionality import get_dimensionality_larsen
# from pymatgen.core.composition import Composition

from mattermake.data.crystal_tokenizer import CrystalTokenData
from mattermake.data.crystal_sequence_datamodule import CrystalSequenceDataModule
from mattermake.models.hierarchical_crystal_transformer_module import (
    HierarchicalCrystalTransformerModule,
)
from mattermake.utils.pylogger import get_pylogger

# Register CrystalTokenData for serialization
torch.serialization.add_safe_globals([CrystalTokenData])

logger = get_pylogger(__name__)


def analyze_structure_properties(structure):
    """
    Analyze structural properties of a generated structure

    Args:
        structure: pymatgen Structure object

    Returns:
        Dict of structural properties
    """
    results = {}

    # Skip invalid structures
    if structure is None:
        return {"valid": False}

    try:
        # Basic structure information
        results["num_sites"] = len(structure)
        results["volume"] = structure.volume
        results["density"] = structure.density
        results["composition"] = structure.composition.reduced_formula
        results["valid"] = True

        # Attempt to get space group
        try:
            sga = SpacegroupAnalyzer(structure, symprec=0.1)
            results["space_group_number"] = sga.get_space_group_number()
            results["space_group_symbol"] = sga.get_space_group_symbol()
        except Exception as e:
            results["space_group_error"] = str(e)

        # Calculate lattice parameters
        lattice = structure.lattice
        results["lattice"] = {
            "a": lattice.a,
            "b": lattice.b,
            "c": lattice.c,
            "alpha": lattice.alpha,
            "beta": lattice.beta,
            "gamma": lattice.gamma,
        }

        # Calculate volume per atom
        results["volume_per_atom"] = structure.volume / len(structure)

        # Try to get dimensionality
        try:
            results["dimensionality"] = get_dimensionality_larsen(structure)
        except Exception as e:
            results["dimensionality_error"] = str(e)

        # Try to get connectivity information
        try:
            vc = VoronoiConnectivity(structure)
            connectivity = vc.get_connectivity_array()
            results["avg_connectivity"] = np.mean([sum(conn) for conn in connectivity])
        except Exception as e:
            results["connectivity_error"] = str(e)

    except Exception as e:
        results["valid"] = False
        results["error"] = str(e)

    return results


def plot_structure_stats(structures_data, output_dir):
    """Plot various statistics of the generated structures"""

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    valid_structures = [s for s in structures_data if s["properties"]["valid"]]
    if not valid_structures:
        logger.info("No valid structures to plot statistics for")
        return

    # Extract properties for valid structures
    volumes = [s["properties"]["volume"] for s in valid_structures]
    densities = [s["properties"]["density"] for s in valid_structures]

    # Get space groups where available
    space_groups = []
    for s in valid_structures:
        if "space_group_number" in s["properties"]:
            space_groups.append(s["properties"]["space_group_number"])

    # Extract lattice parameters
    lattice_a = [s["properties"]["lattice"]["a"] for s in valid_structures]
    lattice_b = [s["properties"]["lattice"]["b"] for s in valid_structures]
    lattice_c = [s["properties"]["lattice"]["c"] for s in valid_structures]
    lattice_alpha = [s["properties"]["lattice"]["alpha"] for s in valid_structures]
    lattice_beta = [s["properties"]["lattice"]["beta"] for s in valid_structures]
    lattice_gamma = [s["properties"]["lattice"]["gamma"] for s in valid_structures]

    # 1. Plot volume distribution
    plt.figure(figsize=(10, 6))
    plt.hist(volumes, bins=20)
    plt.xlabel("Volume (Å³)")
    plt.ylabel("Count")
    plt.title("Volume Distribution of Generated Structures")
    plt.savefig(os.path.join(output_dir, "volume_distribution.png"), dpi=300)
    plt.close()

    # 2. Plot density distribution
    plt.figure(figsize=(10, 6))
    plt.hist(densities, bins=20)
    plt.xlabel("Density (g/cm³)")
    plt.ylabel("Count")
    plt.title("Density Distribution of Generated Structures")
    plt.savefig(os.path.join(output_dir, "density_distribution.png"), dpi=300)
    plt.close()

    # 3. Plot space group distribution if available
    if space_groups:
        unique_sg = sorted(set(space_groups))
        sg_counts = [space_groups.count(sg) for sg in unique_sg]

        plt.figure(figsize=(12, 6))
        plt.bar(unique_sg, sg_counts)
        plt.xlabel("Space Group Number")
        plt.ylabel("Count")
        plt.title("Space Group Distribution of Generated Structures")
        plt.savefig(os.path.join(output_dir, "space_group_distribution.png"), dpi=300)
        plt.close()

    # 4. Plot lattice parameter distributions
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # Lattice lengths
    axs[0, 0].hist(lattice_a, bins=20)
    axs[0, 0].set_xlabel("a (Å)")
    axs[0, 0].set_ylabel("Count")
    axs[0, 0].set_title("Lattice Parameter a")

    axs[0, 1].hist(lattice_b, bins=20)
    axs[0, 1].set_xlabel("b (Å)")
    axs[0, 1].set_title("Lattice Parameter b")

    axs[0, 2].hist(lattice_c, bins=20)
    axs[0, 2].set_xlabel("c (Å)")
    axs[0, 2].set_title("Lattice Parameter c")

    # Lattice angles
    axs[1, 0].hist(lattice_alpha, bins=20)
    axs[1, 0].set_xlabel("α (°)")
    axs[1, 0].set_ylabel("Count")
    axs[1, 0].set_title("Lattice Angle α")

    axs[1, 1].hist(lattice_beta, bins=20)
    axs[1, 1].set_xlabel("β (°)")
    axs[1, 1].set_title("Lattice Angle β")

    axs[1, 2].hist(lattice_gamma, bins=20)
    axs[1, 2].set_xlabel("γ (°)")
    axs[1, 2].set_title("Lattice Angle γ")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "lattice_parameter_distributions.png"), dpi=300
    )
    plt.close()

    logger.info(f"Plots saved to {output_dir}")


def export_structures(structures, output_dir):
    """Export structures to files and analyze properties"""
    os.makedirs(output_dir, exist_ok=True)

    structures_data = []
    valid_count = 0

    for i, structure in enumerate(structures):
        structure_data = {
            "id": i,
            "composition": structure.get("composition", {}),
            "space_group": structure.get("space_group"),
            "properties": {},
        }

        # Export pymatgen structure to CIF if available
        pmg_structure = structure.get("pmg_structure")
        if pmg_structure:
            cif_path = os.path.join(output_dir, f"structure_{i}.cif")
            try:
                pmg_structure.to(filename=cif_path)
                structure_data["cif_path"] = cif_path
                structure_data["properties"] = analyze_structure_properties(
                    pmg_structure
                )
                valid_count += 1
            except Exception as e:
                logger.error(f"Failed to export structure {i} to CIF: {e}")
                structure_data["export_error"] = str(e)
                structure_data["properties"]["valid"] = False
        else:
            structure_data["properties"]["valid"] = False

        structures_data.append(structure_data)

    # Save structure data as JSON
    with open(os.path.join(output_dir, "structures_data.json"), "w") as f:
        json.dump(structures_data, f, indent=2)

    logger.info(f"Exported {valid_count} valid structures to {output_dir}")

    # Plot statistics if we have valid structures
    if valid_count > 0:
        plot_structure_stats(structures_data, output_dir)

    return structures_data


def test_coarse_to_fine_generation(model, tokenizer=None, num_tests=5):
    """Test coarse-to-fine generation by enforcing constraints at each level"""
    logger.info("Testing coarse-to-fine generation with constraints")

    device = next(model.parameters()).device
    results = []

    # Test 1: Generate structures with specific composition
    if tokenizer:
        logger.info("Test 1: Generate with composition constraint")
        for i in range(num_tests):
            try:
                # Create a simple composition constraint (e.g., "Na-Cl")
                comp = ["Na", "Cl"] if i % 2 == 0 else ["Li", "O"]
                tokens = []
                segment_ids = []

                # Add BOS token
                tokens.append(tokenizer.BOS_TOKEN)
                segment_ids.append(tokenizer.SEGMENT_SPECIAL)

                # Add composition tokens
                for element, count in zip(comp, [1, 1]):
                    if (element, count) in tokenizer.composition_tokens:
                        token = tokenizer.composition_tokens[(element, count)]
                        tokens.append(token)
                        segment_ids.append(tokenizer.SEGMENT_COMPOSITION)

                # Add composition separator
                tokens.append(tokenizer.COMP_SEP_TOKEN)
                segment_ids.append(tokenizer.SEGMENT_SPECIAL)

                # Convert to tensors
                start_tokens = torch.tensor([tokens], device=device)
                start_segments = torch.tensor([segment_ids], device=device)

                # Generate with constraint
                generated = model.generate_structure(
                    start_tokens=start_tokens,
                    start_segments=start_segments,
                    temperature=0.8,
                    top_k=40,
                    top_p=0.9,
                )[0]

                results.append(
                    {
                        "test": "composition",
                        "constraint": "-".join(comp),
                        "result": generated.get("composition", {}),
                        "success": all(
                            element in generated.get("composition", {})
                            for element in comp
                        ),
                        "structure": generated,
                    }
                )

            except Exception as e:
                logger.error(f"Error in composition constraint test {i}: {e}")
                results.append(
                    {
                        "test": "composition",
                        "constraint": "-".join(comp)
                        if "comp" in locals()
                        else "unknown",
                        "error": str(e),
                        "success": False,
                    }
                )

    # Test 2: Generate structures with fixed space group
    logger.info("Test 2: Generate with space group constraint")
    for i in range(num_tests):
        try:
            # Try a high symmetry space group
            space_group = 225 if i % 2 == 0 else 194  # Cubic Fm-3m or Hexagonal P63/mmc

            if tokenizer:
                tokens = []
                segment_ids = []

                # Add BOS token
                tokens.append(tokenizer.BOS_TOKEN)
                segment_ids.append(tokenizer.SEGMENT_SPECIAL)

                # Get space group token
                sg_token = tokenizer.vocab.get(f"SG_{space_group}")
                if sg_token:
                    # Skip composition and go straight to space group
                    tokens.append(tokenizer.COMP_SEP_TOKEN)
                    segment_ids.append(tokenizer.SEGMENT_SPECIAL)

                    tokens.append(sg_token)
                    segment_ids.append(tokenizer.SEGMENT_SPACE_GROUP)

                    # Convert to tensors
                    start_tokens = torch.tensor([tokens], device=device)
                    start_segments = torch.tensor([segment_ids], device=device)

                    # Generate with constraint
                    generated = model.generate_structure(
                        start_tokens=start_tokens,
                        start_segments=start_segments,
                        temperature=0.7,
                        top_k=40,
                        top_p=0.9,
                    )[0]

                    results.append(
                        {
                            "test": "space_group",
                            "constraint": space_group,
                            "result": generated.get("space_group"),
                            "success": generated.get("space_group") == space_group,
                            "structure": generated,
                        }
                    )
            else:
                # Without tokenizer, we can't properly construct the input
                logger.warning("Cannot test space group constraint without tokenizer")

        except Exception as e:
            logger.error(f"Error in space group constraint test {i}: {e}")
            results.append(
                {
                    "test": "space_group",
                    "constraint": space_group
                    if "space_group" in locals()
                    else "unknown",
                    "error": str(e),
                    "success": False,
                }
            )

    # Summarize results
    logger.info("\nCoarse-to-fine generation test results:")
    for test_type in ["composition", "space_group"]:
        test_results = [r for r in results if r["test"] == test_type]
        if test_results:
            success_rate = sum(
                1 for r in test_results if r.get("success", False)
            ) / len(test_results)
            logger.info(
                f"{test_type.capitalize()} constraint tests: {success_rate * 100:.1f}% success rate"
            )

            for i, result in enumerate(
                [r for r in test_results if r.get("success", False)]
            ):
                if i < 3:  # Show first 3 successful examples
                    logger.info(
                        f"  Success example: Constraint={result['constraint']}, Result={result['result']}"
                    )

    return results


def evaluate_conditional_generation(model, data_module, num_samples=10):
    """Evaluate conditional generation by taking examples from the dataset"""
    logger.info("\nEvaluating conditional generation from dataset examples")

    # Get validation dataloader
    val_loader = data_module.val_dataloader()
    device = next(model.parameters()).device

    # Get a batch of data
    batch = next(iter(val_loader))

    results = []
    tokenizer = data_module.get_tokenizer()

    for i in range(min(num_samples, len(batch["input_ids"]))):
        try:
            # Get sample from batch
            input_ids = batch["input_ids"][i].to(device)
            segment_ids = batch["segment_ids"][i].to(device)

            # Extract composition section for conditioning
            comp_mask = segment_ids == tokenizer.SEGMENT_COMPOSITION
            comp_sep_mask = input_ids == tokenizer.COMP_SEP_TOKEN

            # Find the position after all composition tokens and before COMP_SEP
            comp_positions = torch.nonzero(comp_mask).squeeze(-1)
            comp_sep_position = torch.nonzero(comp_sep_mask).squeeze(-1)[0]

            if len(comp_positions) > 0 and comp_sep_position > 0:
                # Get just the composition part plus BOS token
                condition_length = comp_sep_position + 1  # Include the COMP_SEP token
                condition_ids = input_ids[:condition_length].unsqueeze(0)
                condition_segments = segment_ids[:condition_length].unsqueeze(0)

                # Get the original structure details for comparison
                original_material_id = batch["material_id"][i]
                original_composition = batch["composition"][i]

                # Generate structure based on composition
                generated = model.generate_structure(
                    start_tokens=condition_ids,
                    start_segments=condition_segments,
                    temperature=0.7,
                    top_k=40,
                    top_p=0.9,
                )[0]

                # Compare with original
                generated_comp = generated.get("composition", {})
                match = all(
                    element in generated_comp for element in original_composition
                )

                results.append(
                    {
                        "material_id": original_material_id,
                        "original_composition": original_composition,
                        "generated_composition": generated_comp,
                        "composition_match": match,
                        "structure": generated,
                        "success": match and "pmg_structure" in generated,
                    }
                )

                logger.info(f"Sample {i}: Material ID={original_material_id}")
                logger.info(f"  Original composition: {original_composition}")
                logger.info(f"  Generated composition: {generated_comp}")
                logger.info(f"  Match: {'Yes' if match else 'No'}")
                logger.info(
                    f"  Valid structure: {'Yes' if 'pmg_structure' in generated else 'No'}"
                )

        except Exception as e:
            logger.error(f"Error in conditional generation sample {i}: {e}")

    # Compute success rate
    success_count = sum(1 for r in results if r.get("success", False))
    success_rate = success_count / len(results) if results else 0
    logger.info(
        f"\nConditional generation success rate: {success_rate * 100:.1f}% ({success_count}/{len(results)})"
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Crystal Transformer Model")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory with processed data"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="generated_structures",
        help="Output directory",
    )
    parser.add_argument(
        "--num_structures",
        type=int,
        default=20,
        help="Number of structures to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature"
    )
    parser.add_argument(
        "--conditional_tests",
        type=int,
        default=10,
        help="Number of conditional generation tests",
    )
    parser.add_argument(
        "--constraint_tests",
        type=int,
        default=5,
        help="Number of constraint tests per type",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up data module
    logger.info(f"Loading data from {args.data_dir}")
    data_module = CrystalSequenceDataModule(
        data_dir=args.data_dir,
        batch_size=16,
        num_workers=4,
    )

    # Prepare data module
    data_module.prepare_data()
    data_module.setup(stage="fit")
    tokenizer = data_module.get_tokenizer()

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    model = HierarchicalCrystalTransformerModule.load_from_checkpoint(args.checkpoint)

    # Set tokenizer config in the model
    if tokenizer:
        model.tokenizer_config = {
            "idx_to_token": tokenizer.idx_to_token,
            "token_to_idx": tokenizer.vocab,
            "lattice_bins": tokenizer.lattice_bins,
            "coordinate_precision": tokenizer.coordinate_precision,
        }

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 1. Generate unconstrained structures
    logger.info(f"\nGenerating {args.num_structures} unconstrained structures")
    with torch.no_grad():
        structures = model.generate_structure(
            num_return_sequences=args.num_structures,
            temperature=args.temperature,
            top_k=40,
            top_p=0.9,
        )

    # Export and analyze structures
    unconstrained_dir = os.path.join(args.output_dir, "unconstrained")
    structures_data = export_structures(structures, unconstrained_dir)

    # 2. Test coarse-to-fine generation with constraints
    with torch.no_grad():
        constraint_results = test_coarse_to_fine_generation(
            model, tokenizer=tokenizer, num_tests=args.constraint_tests
        )

    # Export successful constrained structures
    constrained_dir = os.path.join(args.output_dir, "constrained")
    valid_structures = [
        r["structure"]
        for r in constraint_results
        if r.get("success", False) and "structure" in r
    ]
    if valid_structures:
        export_structures(valid_structures, constrained_dir)

    # 3. Evaluate conditional generation from dataset examples
    with torch.no_grad():
        conditional_results = evaluate_conditional_generation(
            model, data_module, num_samples=args.conditional_tests
        )

    # Export successful conditional structures
    conditional_dir = os.path.join(args.output_dir, "conditional")
    valid_conditional = [
        r["structure"]
        for r in conditional_results
        if r.get("success", False) and "structure" in r
    ]
    if valid_conditional:
        export_structures(valid_conditional, conditional_dir)

    # Summarize overall results
    logger.info("\n===== SUMMARY =====")
    # Unconstrained generation
    valid_count = sum(1 for s in structures_data if s["properties"]["valid"])
    logger.info(
        f"Unconstrained generation: {valid_count}/{args.num_structures} valid structures"
    )

    # Constrained generation
    for test_type in ["composition", "space_group"]:
        test_results = [r for r in constraint_results if r["test"] == test_type]
        if test_results:
            success_rate = sum(
                1 for r in test_results if r.get("success", False)
            ) / len(test_results)
            logger.info(
                f"{test_type.capitalize()} constraint: {success_rate * 100:.1f}% success rate"
            )

    # Conditional generation
    success_count = sum(1 for r in conditional_results if r.get("success", False))
    logger.info(
        f"Conditional generation: {success_count}/{len(conditional_results)} successful generations"
    )


if __name__ == "__main__":
    main()

    # Usage:
    # python -m mattermake.scripts.evaluate_crystal_transformer \
    #     --data_dir processed_data \
    #     --checkpoint path/to/checkpoint.ckpt \
    #     --output_dir generated_structures \
    #     --num_structures 20 \
    #     --temperature 0.8 \
    #     --conditional_tests 10 \
    #     --constraint_tests 5
