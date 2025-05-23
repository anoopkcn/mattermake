import pandas as pd
import torch
from tqdm import tqdm
import warnings
from pathlib import Path
from typing import Dict, Any, Optional

from mattermake.data.components.cif_processing import (
    parse_cif_string,
    get_composition_vector,
    get_spacegroup_number,
    get_lattice_parameters,
    get_asymmetric_unit_atoms,
)
from mattermake.data.components.wyckoff_interface import wyckoff_tuple_to_index
from mattermake.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

warnings.filterwarnings(
    "ignore", message=".*fractional coordinates rounded to ideal values.*"
)

VOCAB_SIZE = 100  # Number of elements to support in composition vector


def process_structure(
    material_id: str,
    cif_string: str,
    vocab_size: int = VOCAB_SIZE,
) -> Optional[Dict[str, Any]]:
    """Process a single structure and return a dictionary of its properties."""
    try:
        structure = parse_cif_string(cif_string)
        comp_vec = get_composition_vector(structure, vocab_size)
        sg_num = get_spacegroup_number(structure)
        lattice_params = get_lattice_parameters(structure)
        atom_seq = get_asymmetric_unit_atoms(structure)
        atom_types = [item[0] for item in atom_seq]
        atom_wyckoffs = [item[1] for item in atom_seq]
        atom_coords = [item[2] for item in atom_seq]

        # Convert Wyckoff letters to global indices
        wyckoff_indices = []
        for wyckoff_letter in atom_wyckoffs:
            if isinstance(wyckoff_letter, str):
                idx = wyckoff_tuple_to_index(sg_num, wyckoff_letter)
                wyckoff_indices.append(idx)
            else:
                # Handle numeric Wyckoff positions (legacy format)
                wyckoff_indices.append(0)  # Use padding index

        return {
            "material_id": material_id,
            "composition": torch.tensor(comp_vec, dtype=torch.int64),
            "spacegroup": torch.tensor(sg_num, dtype=torch.int64),
            "lattice": torch.tensor(lattice_params, dtype=torch.float32),
            "atom_types": torch.tensor(atom_types, dtype=torch.int64),
            "wyckoff": torch.tensor(wyckoff_indices, dtype=torch.int64),
            "atom_coords": torch.tensor(atom_coords, dtype=torch.float32),
        }
    except Exception as e:
        log.error(f"Error processing {material_id}: {str(e)}")
        return None


def process_dataframe(
    df: pd.DataFrame,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    vocab_size: int = VOCAB_SIZE,
):
    """Process dataframe and split into train/val/test sets, saving a single file per split."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Ratios must sum to 1"
    )

    log.info(f"Processing dataframe with {len(df)} entries")
    log.info(f"Output will be saved to {output_dir}")
    log.info(f"Split ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Shuffle and split the dataframe
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_samples = len(df)

    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    log.info(
        f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )

    # Process each split
    splits = [("train", train_df), ("val", val_df), ("test", test_df)]

    for split_name, split_df in splits:
        log.info(f"Processing {split_name} set with {len(split_df)} structures")

        # Create a list to hold all processed structures
        all_structures = []
        processed_count = 0

        # Process each structure
        for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
            material_id = row["material_id"]
            cif_string = row["cif"]
            processed = process_structure(material_id, cif_string, vocab_size)

            if processed:
                all_structures.append(processed)
                processed_count += 1

                # Log progress every 100 structures
                if processed_count % 100 == 0:
                    log.info(
                        f"Processed {processed_count}/{len(split_df)} structures in {split_name} set"
                    )

        # Save all structures for this split to a single file
        output_file = output_path / f"{split_name}.pt"
        log.info(f"Saving {len(all_structures)} processed structures to {output_file}")
        torch.save(all_structures, output_file)

        log.info(
            f"Finished processing {split_name} set. Total processed: {len(all_structures)}"
        )

    return {"train": len(train_df), "val": len(val_df), "test": len(test_df)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Input CSV file with columns material_id,cif",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for processed data, will create train/val/test subdirectories",
    )
    parser.add_argument(
        "--vocab_size", type=int, default=VOCAB_SIZE, help="Element vocabulary size"
    )
    parser.add_argument(
        "--num_structures",
        type=int,
        default=None,
        help="number of structures to be processed",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of data to use for training",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Ratio of data to use for validation",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Ratio of data to use for testing",
    )
    # Batch size parameter removed as we're saving individual files
    args = parser.parse_args()

    log.info("Starting data preparation process")
    log.info(f"Input CSV: {args.input_csv}")
    log.info(f"Output directory: {args.output_dir}")
    log.info(f"Vocabulary size: {args.vocab_size}")

    # Validate split ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        log.warning(f"Split ratios sum to {total_ratio}, not 1.0. Normalizing...")
        args.train_ratio /= total_ratio
        args.val_ratio /= total_ratio
        args.test_ratio /= total_ratio

    if args.num_structures is not None:
        log.info(f"Number of structures to be processed: {args.num_structures}")

    df = pd.read_csv(args.input_csv)
    log.info(f"Loaded {len(df)} entries from CSV file")
    if args.num_structures is not None:
        df = df.head(args.num_structures)

    process_dataframe(
        df,
        args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        vocab_size=args.vocab_size,
    )
