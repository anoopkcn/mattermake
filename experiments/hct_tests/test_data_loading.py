import os
import argparse
import torch.serialization
from mattermake.data.hct_tokenizer import CrystalTokenData
from mattermake.data.hct_sequence_datamodule import CrystalSequenceDataModule

torch.serialization.add_safe_globals([CrystalTokenData])


def analyze_batch(batch, tokenizer=None):
    """Analyze contents of a batch"""
    print(f"Batch keys: {batch.keys()}")
    print(f"Batch size: {len(batch['input_ids'])}")
    print(f"Input shape: {batch['input_ids'].shape}")
    print(f"Segment IDs shape: {batch['segment_ids'].shape}")
    print(f"Target shape: {batch['target_ids'].shape}")

    # Print example sequence
    example_idx = 0
    material_id = batch["material_id"][example_idx]

    print(f"\nExample ({material_id}):")
    input_ids = batch["input_ids"][example_idx]
    segment_ids = batch["segment_ids"][example_idx]

    # Find first padding token
    try:
        seq_len = input_ids.tolist().index(2)  # 2 is PAD_TOKEN
    except ValueError:
        seq_len = len(input_ids)

    print(f"Actual sequence length: {seq_len}")

    # Print sequence with segment types
    if tokenizer:
        print("\nToken sequence:")
        print("-" * 80)
        segment_names = [
            "SPECIAL",
            "COMPOSITION",
            "SPACE_GROUP",
            "LATTICE",
            "ELEMENT",
            "WYCKOFF",
            "COORDINATE",
        ]

        for i in range(min(seq_len, 50)):  # Print first 50 tokens
            token_id = input_ids[i].item()
            segment_id = segment_ids[i].item()
            token_name = tokenizer.idx_to_token.get(token_id, f"UNK-{token_id}")
            segment_name = (
                segment_names[segment_id]
                if segment_id < len(segment_names)
                else f"UNK-{segment_id}"
            )

            print(f"{i:3d} | {token_id:5d} | {token_name:20s} | {segment_name}")

        if seq_len > 50:
            print("... (truncated)")
        print("-" * 80)

    return batch


def test_data_loading(data_dir, batch_size=4, max_length=None, num_workers=0):
    """Test data loading with the CrystalSequenceDataModule"""
    print(f"Testing data loading from {data_dir}")

    # Initialize data module
    data_module = CrystalSequenceDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        max_sequence_length=max_length,
        cache_tokenized=True,
    )

    # Set up data module
    data_module.prepare_data()

    try:
        data_module.setup(stage="fit")
        print("Successfully set up data module")
    except Exception as e:
        print(f"Error setting up data module: {e}")
        return None

    # Print dataset statistics
    data_module.print_dataset_statistics()

    # Get dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    print(f"Training dataloader contains {len(train_loader)} batches")
    print(f"Validation dataloader contains {len(val_loader)} batches")

    # Get tokenizer
    tokenizer = data_module.get_tokenizer()

    # Test batch loading
    print("\n=== Testing batch loading ===")
    train_batch = next(iter(train_loader))
    analyze_batch(train_batch, tokenizer)

    # Check memory usage
    import psutil

    process = psutil.Process(os.getpid())
    memory_gb = process.memory_info().rss / (1024 * 1024 * 1024)
    print(f"\nCurrent memory usage: {memory_gb:.2f} GB")

    return data_module


def main():
    parser = argparse.ArgumentParser(description="Test crystal data loading")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory with processed data"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--max_length", type=int, default=None, help="Max sequence length"
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of dataloader workers"
    )
    args = parser.parse_args()

    test_data_loading(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
    # Usage:
    # python -m mattermake.scripts.test_data_loading \
    #     --data_dir processed_data \
    #     --batch_size 8 \
    #     --max_length 512 \
    #     --num_workers 2
