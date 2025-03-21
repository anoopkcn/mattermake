import os
import torch
from typing import Optional, List, Tuple
import csv
from pymatgen.core import Structure

from mattermake.utils.crystal_to_graph import structure_to_quotient_graph
from mattermake.utils.pylogger import get_pylogger
from mattermake.data.crystal_datamodule import CrystalGraphData

log = get_pylogger(__name__)


def parse_cif_to_structure(
    cif_string: str, material_id: str
) -> Optional[CrystalGraphData]:
    """Parse CIF string to graph components with error handling."""
    try:
        structure = Structure.from_str(cif_string, fmt="cif")
        node_features, edge_index, edge_features, cell_params = (
            structure_to_quotient_graph(structure)
        )

        return CrystalGraphData(
            material_id=material_id,
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            cell_params=cell_params,
            num_nodes=node_features.size(0),
        )
    except Exception as e:
        log.warning(f"Error parsing CIF for {material_id}: {e}")
        return None


def process_batch(
    material_ids: List[str], cif_strings: List[str]
) -> List[CrystalGraphData]:
    """Process a batch of CIF strings to CrystalGraphData objects."""
    results = []
    for material_id, cif_string in zip(material_ids, cif_strings):
        result = parse_cif_to_structure(cif_string, material_id)
        if result is not None:
            results.append(result)
    return results


def load_csv_data(input_path: str, data_limit: float) -> List[Tuple[str, str]]:
    """Load data from CSV file."""
    with open(input_path, "r") as f:
        reader = csv.DictReader(f)
        data = [(row["material_id"], row["cif"]) for row in reader]

    if data_limit < 1.0:
        limit_samples = int(len(data) * data_limit)
        log.info(
            f"Limiting data to {data_limit:.1%}: {limit_samples}/{len(data)} samples"
        )
        data = data[:limit_samples]

    return data


def split_data(
    data: List[CrystalGraphData], train_ratio: float, seed: int
) -> Tuple[List[CrystalGraphData], List[CrystalGraphData], List[CrystalGraphData]]:
    """Split data into train, validation and test sets."""
    torch.manual_seed(seed)
    n = len(data)
    indices = torch.randperm(n).tolist()

    train_size = int(n * train_ratio)
    val_size = int(n * (1 - train_ratio) / 2)

    train_data = [data[i] for i in indices[:train_size]]
    val_data = [data[i] for i in indices[train_size : train_size + val_size]]
    test_data = [data[i] for i in indices[train_size + val_size :]]

    return train_data, val_data, test_data


def prepare_structure_data(
    input_path: str,
    output_dir: str,
    train_ratio: float = 0.8,
    data_limit: float = 1.0,
    seed: int = 42,
    batch_size: int = 1000,
):
    raw_data = load_csv_data(input_path, data_limit)

    all_graphs = []
    for i in range(0, len(raw_data), batch_size):
        batch = raw_data[i : i + batch_size]
        material_ids, cif_strings = zip(*batch)
        batch_results = process_batch(material_ids, cif_strings)
        all_graphs.extend(batch_results)

    train_data, val_data, test_data = split_data(all_graphs, train_ratio, seed)

    log.info(f"Train set: {len(train_data)} samples")
    log.info(f"Validation set: {len(val_data)} samples")
    log.info(f"Test set: {len(test_data)} samples")

    os.makedirs(output_dir, exist_ok=True)
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        torch.save(data, os.path.join(output_dir, f"{name}.pt"))

    log.info(f"Data preparation completed and saved to: {output_dir}")


if __name__ == "__main__":
    import argparse

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
