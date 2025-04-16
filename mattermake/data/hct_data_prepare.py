import pandas as pd
import torch
from tqdm import tqdm

from mattermake.data.components.wyckoff_utils import (
    load_wyckoff_symbols,
    wyckoff_symbol_to_index,
)
from mattermake.data.components.cif_processing import (
    parse_cif_string,
    get_composition_vector,
    get_spacegroup_number,
    get_lattice_parameters,
    get_asymmetric_unit_atoms,
)
from mattermake.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


VOCAB_SIZE = 100  # Number of elements to support in composition vector


def process_dataframe(
    df: pd.DataFrame,
    output_path: str,
    vocab_size: int = VOCAB_SIZE,
):
    log.info(f"Processing dataframe with {len(df)} entries")
    log.info(f"Output will be saved to {output_path}")

    sg_to_symbols = load_wyckoff_symbols()
    wyckoff_mapping = wyckoff_symbol_to_index(sg_to_symbols)
    log.info(f"Loaded Wyckoff symbols for {len(sg_to_symbols)} space groups")

    processed_data = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        material_id = row["material_id"]
        cif_string = row["cif"]
        try:
            structure = parse_cif_string(cif_string)
            comp_vec = get_composition_vector(structure, vocab_size)
            sg_num = get_spacegroup_number(structure)
            lattice_params = get_lattice_parameters(structure)
            atom_seq = get_asymmetric_unit_atoms(
                structure, sg_to_symbols, wyckoff_mapping
            )
            atom_types = [item[0] for item in atom_seq]
            atom_wyckoffs = [item[1] for item in atom_seq]
            atom_coords = [item[2] for item in atom_seq]

            processed_data.append(
                {
                    "material_id": material_id,
                    "composition": torch.tensor(comp_vec, dtype=torch.int64),
                    "spacegroup": torch.tensor(sg_num, dtype=torch.int64),
                    "lattice": torch.tensor(lattice_params, dtype=torch.float32),
                    "atom_types": torch.tensor(atom_types, dtype=torch.int64),
                    "atom_wyckoffs": torch.tensor(atom_wyckoffs, dtype=torch.int64),
                    "atom_coords": torch.tensor(atom_coords, dtype=torch.float32),
                }
            )
            if len(processed_data) % 100 == 0:
                log.debug(f"Processed {len(processed_data)} structures")

        except Exception as e:
            log.error(f"Error processing {material_id}: {str(e)}")

    # Save processed data as a torch file
    torch.save(processed_data, output_path)
    log.info(f"Saved {len(processed_data)} processed structures to {output_path}")


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
        "--wyckoff_symbols", type=str, required=True, help="Path to wyckoff_symbols.csv"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output .pt file for processed data"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=VOCAB_SIZE, help="Element vocabulary size"
    )
    args = parser.parse_args()

    log.info("Starting data preparation process")
    log.info(f"Input CSV: {args.input_csv}")
    log.info(f"Output file: {args.output}")
    log.info(f"Vocabulary size: {args.vocab_size}")

    df = pd.read_csv(args.input_csv)
    log.info(f"Loaded {len(df)} entries from CSV file")

    process_dataframe(df, args.output, args.vocab_size)
