import os
import torch
import argparse
import warnings
import pandas as pd
from tqdm import tqdm
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from mattermake.data.crystal_tokenizer import CrystalTokenizer

from mattermake.utils.pylogger import get_pylogger

# Suppress pymatgen warnings about fractional coordinates
warnings.filterwarnings("ignore", message=".*fractional coordinates rounded to ideal values.*")

logger = get_pylogger(__name__)


def process_structure_from_cif_string(
    material_id, cif_string, tokenizer, standardize=True, symprec=0.01
):
    """Process a structure from a CIF string"""
    try:
        structure = Structure.from_str(cif_string, fmt="cif")

        if standardize:
            sga = SpacegroupAnalyzer(structure, symprec=symprec)
            structure = sga.get_conventional_standard_structure()
            space_group = sga.get_space_group_number()
        else:
            space_group = None

        token_data = tokenizer.tokenize_structure(structure)

        return {
            "material_id": material_id,
            "formula": structure.composition.reduced_formula,
            "space_group": space_group,
            "token_data": token_data,
            "structure": structure,
        }
    except Exception as e:
        logger.error(f"Error processing material {material_id}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Process crystal structures from CSV with CIF strings"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="CSV file with material_id and cif columns",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=256, help="Maximum sequence length"
    )
    parser.add_argument(
        "--coord_precision", type=int, default=4, help="Coordinate precision"
    )
    parser.add_argument(
        "--split_ratio", type=float, default=0.1, help="Validation split ratio"
    )
    parser.add_argument(
        "--standardize", action="store_true", help="Standardize structures"
    )
    parser.add_argument(
        "--symprec", type=float, default=0.01, help="Symmetry precision"
    )
    parser.add_argument(
        "--material_id_col",
        type=str,
        default="material_id",
        help="Column name for material IDs",
    )
    parser.add_argument(
        "--cif_col", type=str, default="cif", help="Column name for CIF strings"
    )
    parser.add_argument(
        "--max_structures",
        type=int,
        default=None,
        help="Maximum number of structures to process (for testing)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = CrystalTokenizer(
        max_sequence_length=args.max_seq_length,
        coordinate_precision=args.coord_precision,
    )

    logger.info(f"Loading CSV file: {args.input_csv}")
    df = pd.read_csv(args.input_csv)

    if args.material_id_col not in df.columns:
        raise ValueError(
            f"Material ID column '{args.material_id_col}' not found in CSV"
        )
    if args.cif_col not in df.columns:
        raise ValueError(f"CIF column '{args.cif_col}' not found in CSV")

    logger.info(f"Found {len(df)} entries in CSV")

    # Limit number of structures if requested
    if args.max_structures is not None and args.max_structures > 0:
        logger.info(f"Limiting to {args.max_structures} structures for testing")
        df = df.head(args.max_structures)

    processed_data = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing structures"):
        material_id = row[args.material_id_col]
        cif_string = row[args.cif_col]

        result = process_structure_from_cif_string(
            material_id,
            cif_string,
            tokenizer,
            standardize=args.standardize,
            symprec=args.symprec,
        )

        if result:
            processed_data.append(result)

    stats = {
        "total_structures": len(processed_data),
        "unique_formulas": len(set(item["formula"] for item in processed_data)),
        "unique_space_groups": len(
            set(item["space_group"] for item in processed_data if item["space_group"])
        ),
    }

    logger.info(f"Successfully processed {stats['total_structures']} structures")

    from sklearn.model_selection import train_test_split

    strata = [
        item["space_group"] % 10 if item["space_group"] else 0
        for item in processed_data
    ]

    train_data, val_data = train_test_split(
        processed_data,
        test_size=args.split_ratio,
        random_state=42,
        stratify=strata,
    )

    logger.info(
        f"Split data into {len(train_data)} training and {len(val_data)} validation structures"
    )

    save_path = os.path.join(args.output_dir, "processed_crystal_data.pt")
    logger.info(f"Saving processed data to {save_path}")

    torch.save(
        {
            "train_data": train_data,
            "val_data": val_data,
            "tokenizer_config": {
                "max_sequence_length": args.max_seq_length,
                "coordinate_precision": args.coord_precision,
                "lattice_bins": tokenizer.lattice_bins,
                # "vocab_size": tokenizer.vocab_size,
            },
            "stats": stats,
        },
        save_path,
    )

    summary = []
    for item in processed_data:
        summary.append(
            {
                "material_id": item["material_id"],
                "formula": item["formula"],
                "space_group": item["space_group"],
                "seq_length": len(item["token_data"].sequence),
                "num_atoms": len(item["structure"]),
            }
        )

    summary_path = os.path.join(args.output_dir, "data_summary.csv")
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    logger.info(f"Saved data summary to {summary_path}")

    logger.info("\nDataset Statistics:")
    logger.info(f"Total structures: {stats['total_structures']}")
    logger.info(f"Unique formulas: {stats['unique_formulas']}")
    logger.info(f"Unique space groups: {stats['unique_space_groups']}")


if __name__ == "__main__":
    main()

    # Usage::
    # python -m mattermake.scripts.prepare_crystal_data \
    # --input_csv path/to/your/structures.csv \
    # --output_dir processed_data \
    # --material_id_col material_id \
    # --cif_col cif \
    # --max_seq_length 512 \
    # --max_structures 1000
    # --standardize
