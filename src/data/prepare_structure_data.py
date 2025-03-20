import os
import torch
import pandas as pl
import argparse

from pymatgen.core import Structure

from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def prepare_structure_data(
    input_path: str,
    output_dir: str,
    train_ratio: float = 0.8,
    data_limit: float = 1.0,
    seed: int = 42,
):
    log.info(f"Loading data from {input_path}")
    val_ratio = (1.0 - train_ratio) / 2.0
    test_ratio = val_ratio

    log.info(
        f"Train ratio: {train_ratio:.1%}, Validation ratio: {val_ratio:.1%}, Test ratio: {test_ratio:.1%}"
    )

    data = pl.read_csv(input_path, null_values=["nan"])
    data["structures"] = data["cif"].apply(lambda x: Structure.from_str(x, "cif"))

    data = {"structures": data["structures"]}

    if data_limit < 1.0:
        total_samples = len(data["structures"])
        limit_samples = int(total_samples * data_limit)
        log.info(
            f"Limiting data to {data_limit:.1%} of original size: {limit_samples}/{total_samples} samples"
        )

        data["structures"] = data["structures"][:limit_samples]

    log.info(f"Total number of samples after limiting: {len(data['structures'])}")

    torch.manual_seed(seed)

    indices = torch.randperm(len(data["structures"]))
    data["structures"] = [data["structures"][i.item()] for i in indices]

    n = len(data["structures"])
    train_split_idx = int(n * train_ratio)
    val_split_idx = train_split_idx + int(n * val_ratio)
    test_split_idx = val_split_idx + int(n * test_ratio)

    train_structures = torch.stack(
        [structure for structure in data["structures"][:train_split_idx]]
    )
    val_structures = torch.stack(
        [structure for structure in data["structures"][train_split_idx:val_split_idx]]
    )

    test_structures = torch.stack(
        [structure for structure in data["structures"][val_split_idx:test_split_idx]]
    )

    log.info(f"Train set: {len(train_structures)} samples")
    log.info(f"Validation set: {len(val_structures)} samples")
    log.info(f"Test set: {len(test_structures)} samples")

    log.info("Encoding slices...")

    os.makedirs(output_dir, exist_ok=True)

    train_data = {"structures": train_structures}
    val_data = {"structures": val_structures}
    test_data = {"structures": test_structures}

    train_path = os.path.join(output_dir, "train.pt")
    val_path = os.path.join(output_dir, "val.pt")
    test_path = os.path.join(output_dir, "test.pt")

    torch.save(train_data, train_path)
    torch.save(val_data, val_path)
    torch.save(test_data, test_path)

    log.info("Data preparation completed and saved to:")
    log.info(f"- Train data: {train_path}")
    log.info(f"- Validation data: {val_path}")
    log.info(f"- Test data: {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare slice data for training")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input pickle file with embeddings and slices",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/slice",
        help="Directory to save the processed data",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Ratio of data to use for training (rest for validation)",
    )
    parser.add_argument(
        "--data-limit",
        type=float,
        default=1.0,
        help="Factor to limit the amount of data used (1.0 means use all data, 0.2 means use 20%)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()
    prepare_structure_data(
        args.input, args.output, args.train_ratio, args.data_limit, args.seed
    )
