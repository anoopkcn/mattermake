import os
import argparse
import torch
import json
import numpy as np
from tqdm import tqdm
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.structure_analyzer import VoronoiConnectivity

from mattermake.models.hierarchical_crystal_transformer_module import (
    CrystalTransformerModule,
)
from mattermake.data.crystal_tokenizer import CrystalTokenizer
from mattermake.utils.crystal_sequence_utils import CrystalSequenceDecoder
from mattermake.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def load_test_data(data_path):
    """Load test structures"""
    log.info(f"Loading test data from {data_path}")
    test_data = torch.load(data_path)
    return test_data


def evaluate_structures(generated_structures, reference_structures):
    """Evaluate generated structures against reference structures"""
    log.info("Evaluating generated structures")

    # Initialize structure matcher
    matcher = StructureMatcher(stol=0.2, ltol=0.2, angle_tol=5.0)

    results = []

    for i, gen_struct_data in enumerate(tqdm(generated_structures, desc="Evaluating")):
        if (
            not gen_struct_data.get("valid", False)
            or "structure" not in gen_struct_data
        ):
            results.append(
                {
                    "index": i,
                    "valid": False,
                    "error": gen_struct_data.get("error", "Invalid structure"),
                }
            )
            continue

        gen_structure = gen_struct_data["structure"]

        # Compute space group
        try:
            sga = SpacegroupAnalyzer(gen_structure)
            space_group = sga.get_space_group_number()
            space_group_symbol = sga.get_space_group_symbol()
        except Exception as e:
            space_group = 0
            space_group_symbol = "Unknown"
            log.warning(f"Error determining space group: {str(e)}")

        # Calculate basic properties
        num_atoms = len(gen_structure)
        volume = gen_structure.volume
        density = gen_structure.density

        # Calculate coordination environment
        try:
            vconn = VoronoiConnectivity(gen_structure)
            connectivity = vconn.connectivity_array

            # Average coordination number
            avg_cn = 0
            for i, site in enumerate(gen_structure):
                cn = np.sum(connectivity[i] > 0)
                avg_cn += cn
            avg_cn /= num_atoms
        except Exception as e:
            avg_cn = 0
            log.warning(f"Error calculating coordination: {str(e)}")

        # Find closest match in reference structures
        closest_match = None
        min_distance = float("inf")

        for ref_data in reference_structures:
            try:
                ref_structure = (
                    Structure.from_dict(ref_data["structure"])
                    if isinstance(ref_data.get("structure"), dict)
                    else ref_data.get("structure")
                )
                if ref_structure:
                    # Check for structure match
                    if matcher.fit(gen_structure, ref_structure):
                        closest_match = ref_data["material_id"]
                        min_distance = 0
                        break
            except Exception as e:
                log.warning(f"Error comparing with reference: {str(e)}")

        # Collect results
        result = {
            "index": i,
            "valid": True,
            "space_group": space_group,
            "space_group_symbol": space_group_symbol,
            "num_atoms": num_atoms,
            "volume": volume,
            "density": density,
            "avg_coordination_number": avg_cn,
            "closest_match": closest_match,
            "min_distance": min_distance,
            "formula": gen_structure.composition.formula,
            "lattice_params": [
                gen_structure.lattice.a,
                gen_structure.lattice.b,
                gen_structure.lattice.c,
                gen_structure.lattice.alpha,
                gen_structure.lattice.beta,
                gen_structure.lattice.gamma,
            ],
        }

        results.append(result)

    return results


def generate_structures(
    model, decoder, num_structures, temperature, max_length, space_group=None
):
    """Generate crystal structures using the model"""
    log.info(f"Generating {num_structures} structures (temperature={temperature})")

    # Prepare starting tokens if space group is specified
    start_tokens = None
    start_segments = None

    if space_group:
        # Create a sequence starting with BOS followed by the specified space group
        sg_token_name = f"SG_{space_group}"
        if sg_token_name in decoder.tokenizer.vocab:
            sg_token_id = decoder.tokenizer.vocab[sg_token_name]
            start_tokens = torch.tensor(
                [[decoder.tokenizer.BOS_TOKEN, sg_token_id]], dtype=torch.long
            )
            start_segments = torch.tensor(
                [
                    [
                        decoder.tokenizer.SEGMENT_SPECIAL,
                        decoder.tokenizer.SEGMENT_SPACE_GROUP,
                    ]
                ],
                dtype=torch.long,
            )
            log.info(f"Starting generation with space group {space_group}")
        else:
            log.warning(
                f"Space group {space_group} not found in tokenizer. Using random space group."
            )

    # Generate token sequences
    generated_outputs = model.generate_structure(
        start_tokens=start_tokens,
        start_segments=start_segments,
        max_length=max_length,
        temperature=temperature,
        top_k=40,
        top_p=0.9,
        num_return_sequences=num_structures,
    )

    # Decode token sequences to structures
    structures = []
    for output in tqdm(generated_outputs, desc="Decoding"):
        tokens = output["tokens"]
        segments = output["segments"]

        # Decode sequence
        structure_data = decoder.decode_sequence(tokens, segments)
        structures.append(structure_data)

    return structures


def calculate_metrics(results):
    """Calculate overall metrics from evaluation results"""
    valid_results = [r for r in results if r.get("valid", False)]

    metrics = {
        "total_structures": len(results),
        "valid_structures": len(valid_results),
        "validity_rate": len(valid_results) / len(results) if results else 0,
        "space_group_distribution": {},
        "average_atoms": 0,
        "average_volume": 0,
        "average_density": 0,
        "average_coordination": 0,
        "novelty_rate": 0,
    }

    if valid_results:
        # Calculate averages
        metrics["average_atoms"] = np.mean([r["num_atoms"] for r in valid_results])
        metrics["average_volume"] = np.mean([r["volume"] for r in valid_results])
        metrics["average_density"] = np.mean([r["density"] for r in valid_results])
        metrics["average_coordination"] = np.mean(
            [r["avg_coordination_number"] for r in valid_results]
        )

        # Calculate novelty rate
        novel_count = sum(1 for r in valid_results if r["closest_match"] is None)
        metrics["novelty_rate"] = novel_count / len(valid_results)

        # Calculate space group distribution
        sg_counts = {}
        for r in valid_results:
            sg = r["space_group"]
            sg_counts[sg] = sg_counts.get(sg, 0) + 1

        for sg, count in sg_counts.items():
            metrics["space_group_distribution"][sg] = count / len(valid_results)

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate crystal structure generation"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--test_data", type=str, required=True, help="Path to test data file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--num_structures",
        type=int,
        default=100,
        help="Number of structures to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature"
    )
    parser.add_argument(
        "--max_length", type=int, default=256, help="Maximum sequence length"
    )
    parser.add_argument(
        "--space_group",
        type=int,
        default=None,
        help="Specify space group to generate (optional)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save_structures",
        action="store_true",
        help="Save generated structures as CIF files",
    )
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load test data
    test_data = load_test_data(args.test_data)
    log.info(f"Loaded {len(test_data)} test structures")

    # Initialize tokenizer from the model checkpoint
    log.info(f"Loading model from checkpoint: {args.checkpoint}")
    model = CrystalTransformerModule.load_from_checkpoint(args.checkpoint)

    # Get tokenizer config from model
    tokenizer_config = model.tokenizer_config

    # Initialize tokenizer with compatible parameters
    tokenizer = CrystalTokenizer(
        max_sequence_length=model.hparams.max_position_embeddings,
        coordinate_precision=4,
        lattice_bins=100,
    )

    # Override tokenizer vocabulary with loaded model's vocabulary
    if tokenizer_config and "token_maps" in tokenizer_config:
        tokenizer.idx_to_token = tokenizer_config["token_maps"]["id_to_token"]
        tokenizer.vocab = {v: k for k, v in tokenizer.idx_to_token.items()}
        tokenizer.element_tokens = tokenizer_config["token_maps"]["element_tokens"]
        tokenizer.wyckoff_tokens = tokenizer_config["token_maps"]["wyckoff_tokens"]
        tokenizer.coord_tokens = tokenizer_config["token_maps"]["coord_tokens"]
        tokenizer.sg_wyckoff_map = tokenizer_config["space_group_data"]
        tokenizer.sg_lattice_constraints = tokenizer_config["lattice_constraints"]

    # Initialize sequence decoder
    decoder = CrystalSequenceDecoder(tokenizer)

    # Generate structures
    generated_structures = generate_structures(
        model=model,
        decoder=decoder,
        num_structures=args.num_structures,
        temperature=args.temperature,
        max_length=args.max_length,
        space_group=args.space_group,
    )

    # Save generated structures if requested
    if args.save_structures:
        from pymatgen.io.cif import CifWriter

        structures_dir = os.path.join(args.output, "structures")
        os.makedirs(structures_dir, exist_ok=True)

        for i, structure_data in enumerate(generated_structures):
            if "structure" in structure_data and structure_data.get("valid", False):
                structure = structure_data["structure"]

                # Generate filename
                sg = structure_data.get("space_group", 0)
                filename = f"generated_{i:03d}_sg{sg}.cif"
                filepath = os.path.join(structures_dir, filename)

                # Save CIF file
                writer = CifWriter(structure)
                writer.write_file(filepath)

    # Evaluate generated structures
    evaluation_results = evaluate_structures(generated_structures, test_data)

    # Calculate overall metrics
    metrics = calculate_metrics(evaluation_results)

    # Save results
    results_path = os.path.join(args.output, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(evaluation_results, f, indent=2)

    metrics_path = os.path.join(args.output, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Generate a summary report
    report = f"""
    # Crystal Transformer Evaluation Report

    ## Generation Settings
    - Model checkpoint: {args.checkpoint}
    - Number of structures: {args.num_structures}
    - Temperature: {args.temperature}
    - Maximum sequence length: {args.max_length}
    - Space group: {args.space_group if args.space_group else "Random"}
    - Seed: {args.seed}

    ## Overall Metrics
    - Total structures generated: {metrics["total_structures"]}
    - Valid structures: {metrics["valid_structures"]} ({metrics["validity_rate"] * 100:.1f}%)
    - Novel structures: {metrics["novelty_rate"] * 100:.1f}%
    - Average number of atoms: {metrics["average_atoms"]:.1f}
    - Average unit cell volume: {metrics["average_volume"]:.1f} Å³
    - Average density: {metrics["average_density"]:.2f} g/cm³
    - Average coordination number: {metrics["average_coordination"]:.2f}

    ## Top 10 Space Groups
    """

    # Add top 10 space groups to the report
    if metrics["space_group_distribution"]:
        top_sgs = sorted(
            metrics["space_group_distribution"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        for sg, percentage in top_sgs:
            report += f"- Space Group {sg}: {percentage * 100:.1f}%\n"

    # Save report
    report_path = os.path.join(args.output, "evaluation_report.md")
    with open(report_path, "w") as f:
        f.write(report)

    log.info(f"Evaluation complete. Results saved to {args.output}")
    log.info(
        f"Valid structures: {metrics['valid_structures']}/{metrics['total_structures']} ({metrics['validity_rate'] * 100:.1f}%)"
    )
    log.info(f"Novel structures: {metrics['novelty_rate'] * 100:.1f}%")


if __name__ == "__main__":
    main()
