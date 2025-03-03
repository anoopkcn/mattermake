import os
import torch
import pandas as pd
import pickle
import argparse

from src.utils.vocab import vocab, encode_slice, stoi, itos
from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

def prepare_slices_data(
    input_path: str,
    output_dir: str,
    train_ratio: float = 0.9,
    data_limit_factor: float = 1.0,
    seed: int = 42
):
    """
    Prepare slice data for training by splitting into train/val sets and encoding slices.

    Args:
        input_path: Path to the input pickle file with embeddings and slices
        output_dir: Directory to save the processed data
        train_ratio: Ratio of data to use for training (rest for validation)
        data_limit_factor: Factor to limit the amount of data used (1.0 means use all data)
        seed: Random seed for reproducibility
    """
    log.info(f"Loading data from {input_path}")

    data = pd.read_pickle(input_path)

    data = {
        'embeddings': data['embeddings'],
        'slices': data['slices']
    }

    # Apply data limiting factor (TODO: change the name of the variable)
    if data_limit_factor < 1.0:
        total_samples = len(data['embeddings'])
        limit_samples = int(total_samples * data_limit_factor)
        log.info(f"Limiting data to {data_limit_factor:.1%} of original size: {limit_samples}/{total_samples} samples")

        data['embeddings'] = data['embeddings'][:limit_samples]
        data['slices'] = data['slices'][:limit_samples]

    embedding_dim = data['embeddings'][0].shape[0]
    log.info(f"Embedding dimension: {embedding_dim}")
    log.info(f"Total number of samples after limiting: {len(data['embeddings'])}")

    torch.manual_seed(seed)

    indices = torch.randperm(len(data['embeddings']))
    data['embeddings'] = [data['embeddings'][i.item()] for i in indices]
    data['slices'] = [data['slices'][i.item()] for i in indices]

    n = len(data['embeddings'])
    split_idx = int(n*train_ratio)

    train_emb = torch.stack([emb.float() for emb in data['embeddings'][:split_idx]])
    val_emb = torch.stack([emb.float() for emb in data['embeddings'][split_idx:]])

    train_slices = data['slices'][:split_idx]
    val_slices = data['slices'][split_idx:]

    log.info(f"Train set: {len(train_emb)} samples")
    log.info(f"Validation set: {len(val_emb)} samples")

    log.info("Encoding slices...")
    train_slice_ids = [encode_slice(slice_text) for slice_text in train_slices]
    val_slice_ids = [encode_slice(slice_text) for slice_text in val_slices]

    os.makedirs(output_dir, exist_ok=True)

    train_data = {
        'embeddings': train_emb,
        'slice_ids': train_slice_ids
    }
    val_data = {
        'embeddings': val_emb,
        'slice_ids': val_slice_ids
    }

    meta = {
        'vocab_size': len(vocab),
        'stoi': stoi,
        'itos': itos,
        'embedding_dim': embedding_dim
    }

    train_path = os.path.join(output_dir, 'train.pt')
    val_path = os.path.join(output_dir, 'val.pt')
    meta_path = os.path.join(output_dir, 'meta.pkl')

    torch.save(train_data, train_path)
    torch.save(val_data, val_path)
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)

    log.info("Data preparation completed and saved to:")
    log.info(f"- Train data: {train_path}")
    log.info(f"- Validation data: {val_path}")
    log.info(f"- Metadata: {meta_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare slice data for training")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input pickle file with embeddings and slices"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/slice",
        help="Directory to save the processed data"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Ratio of data to use for training (rest for validation)"
    )
    parser.add_argument(
        "--data-limit",
        type=float,
        default=1.0,
        help="Factor to limit the amount of data used (1.0 means use all data, 0.2 means use 20%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()
    prepare_slices_data(args.input, args.output, args.train_ratio, args.data_limit, args.seed)
