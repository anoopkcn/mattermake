import os
import torch
import polars as pl
import argparse
import pickle
from typing import Dict, Optional

from pymatgen.core import Structure
from src.utils.crystal_to_graph import structure_to_quotient_graph
from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def parse_cif_to_structure(cif_string: str) -> Optional[Dict]:
    """Parse CIF string to graph components with error handling."""
    try:
        structure = Structure.from_str(cif_string, fmt="cif")
        return structure_to_quotient_graph(structure)
    except Exception as e:
        log.warning(f"Error parsing CIF: {e}")
        return None


def process_batch(cif_strings: list) -> list:
    """Process a batch of CIF strings sequentially."""
    results = []
    for cif_string in cif_strings:
        result = parse_cif_to_structure(cif_string)
        if result is not None:
            node_features, edge_index, edge_features, cell_params = result
            results.append(
                {
                    "node_features": node_features,
                    "edge_index": edge_index,
                    "edge_features": edge_features,
                    "cell_params": cell_params,
                    "num_nodes": node_features.size(0),
                }
            )
    return results


def prepare_structure_data(
    input_path: str,
    output_dir: str,
    train_ratio: float = 0.8,
    data_limit: float = 1.0,
    seed: int = 42,
    batch_size: int = 1000,
):
    data = pl.scan_csv(input_path, null_values=["nan"]).select(["material_id", "cif"])

    if data_limit < 1.0:
        total_samples = data.collect().shape[0]
        limit_samples = int(total_samples * data_limit)
        log.info(
            f"Limiting data to {data_limit:.1%}: {limit_samples}/{total_samples} samples"
        )
        data = data.limit(limit_samples)

    data = data.collect()

    # Process data in batches
    structures = []
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        batch_results = process_batch(batch["cif"].to_list())
        structures.extend(batch_results)

    # Create result dataframe efficiently
    result_data = pl.DataFrame(
        {
            "material_id": data["material_id"],
            "structures": structures,
            "node_features": [s["node_features"] for s in structures],
            "edge_index": [s["edge_index"] for s in structures],
            "edge_features": [s["edge_features"] for s in structures],
            "cell_params": [s["cell_params"] for s in structures],
            "num_nodes": [s["num_nodes"] for s in structures],
        }
    )

    # Split data
    torch.manual_seed(seed)
    n = len(result_data)
    indices = torch.randperm(n).tolist()

    train_size = int(n * train_ratio)
    val_size = int(n * (1 - train_ratio) / 2)

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    train_data = result_data.filter(pl.col("index").is_in(train_indices))
    val_data = result_data.filter(pl.col("index").is_in(val_indices))
    test_data = result_data.filter(pl.col("index").is_in(test_indices))

    # Log info
    log.info(f"Train set: {len(train_data)} samples")
    log.info(f"Validation set: {len(val_data)} samples")
    log.info(f"Test set: {len(test_data)} samples")

    # Save data
    os.makedirs(output_dir, exist_ok=True)
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        with open(os.path.join(output_dir, f"{name}.pkl"), "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    log.info(f"Data preparation completed and saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare structure data for training")
    parser.add_argument("--input", type=str, required=True, help="Input CSV path")
    parser.add_argument(
        "--output", type=str, default="data/structures", help="Output directory"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="Training data ratio"
    )
    parser.add_argument(
        "--data-limit", type=float, default=1.0, help="Data limit factor"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--batch-size", type=int, default=1000, help="Processing batch size"
    )

    args = parser.parse_args()
    prepare_structure_data(
        args.input,
        args.output,
        args.train_ratio,
        args.data_limit,
        args.seed,
        args.batch_size,
    )
