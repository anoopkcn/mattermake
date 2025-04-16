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

        return {
            "material_id": material_id,
            "composition": torch.tensor(comp_vec, dtype=torch.int64),
            "spacegroup": torch.tensor(sg_num, dtype=torch.int64),
            "lattice": torch.tensor(lattice_params, dtype=torch.float32),
            "atom_types": torch.tensor(atom_types, dtype=torch.int64),
            "atom_wyckoffs": torch.tensor(atom_wyckoffs, dtype=torch.int64),
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
    batch_size: int = 1000,
    vocab_size: int = VOCAB_SIZE,
):
    """Process dataframe and split into train/val/test sets."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    log.info(f"Processing dataframe with {len(df)} entries")
    log.info(f"Output will be saved to {output_dir}")
    log.info(f"Split ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    
    # Create output directories
    output_path = Path(output_dir)
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    test_dir = output_path / "test"
    
    for dir_path in [train_dir, val_dir, test_dir]:
        dir_path.mkdir(exist_ok=True, parents=True)
    
    # Shuffle and split the dataframe
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_samples = len(df)
    
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    log.info(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    # Process each split
    splits = [
        ("train", train_df, train_dir),
        ("validation", val_df, val_dir),
        ("test", test_df, test_dir)
    ]
    
    for split_name, split_df, split_dir in splits:
        log.info(f"Processing {split_name} set with {len(split_df)} structures")
        
        # Process in batches to avoid memory issues with large datasets
        total_processed = 0
        for batch_idx in range(0, len(split_df), batch_size):
            batch_df = split_df.iloc[batch_idx:batch_idx + batch_size]
            batch_data = []
            
            # Process each structure in this batch
            for _, row in tqdm(batch_df.iterrows(), total=len(batch_df)):
                material_id = row["material_id"]
                cif_string = row["cif"]
                processed = process_structure(material_id, cif_string, vocab_size)
                if processed:
                    batch_data.append(processed)
            
            if batch_data:
                batch_file = split_dir / f"batch_{batch_idx // batch_size}.pt"
                torch.save(batch_data, batch_file)
                log.info(f"Saved {len(batch_data)} structures to {batch_file}")
                total_processed += len(batch_data)
        
        log.info(f"Finished processing {split_name} set. Total processed: {total_processed}")
    
    return {
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df)
    }


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
        help="Output directory for processed data, will create train/val/test subdirectories"
    )
    parser.add_argument(
        "--vocab_size", 
        type=int, 
        default=VOCAB_SIZE, 
        help="Element vocabulary size"
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Number of structures to process in each batch",
    )
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
        batch_size=args.batch_size,
        vocab_size=args.vocab_size
    )
