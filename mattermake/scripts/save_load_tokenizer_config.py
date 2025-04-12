from mattermake.data.crystal_sequence_datamodule import CrystalSequenceDataModule


def save_tokenizer_config(data_dir, output_path):
    data_module = CrystalSequenceDataModule(data_dir=data_dir, batch_size=1)
    data_module.prepare_data()
    data_module.setup(stage="fit")

    tokenizer = data_module.get_tokenizer()

    if tokenizer:
        tokenizer_config = {
            "idx_to_token": tokenizer.idx_to_token,
            "token_to_idx": tokenizer.vocab,
            "lattice_bins": getattr(tokenizer, "lattice_bins", None),
            "coordinate_precision": getattr(tokenizer, "coordinate_precision", None),
        }

        import json

        with open(output_path, "w") as f:
            json.dump(tokenizer_config, f)

        print(f"Tokenizer config saved to {output_path}")

    else:
        print("No tokenizer found")


def load_tokenizer_config(config_path):
    import json

    with open(config_path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Save or load tokenizer configuration")
    parser.add_argument("data_dir", help="Directory containing processed data for tokenization")
    parser.add_argument("output_path", help="Path to save tokenizer configuration JSON")
    
    args = parser.parse_args()
    
    # Save tokenizer config to the specified output path
    save_tokenizer_config(args.data_dir, args.output_path)
