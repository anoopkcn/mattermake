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

VOCAB_SIZE = 100  # Number of elements to support in composition vector


def process_dataframe(
    df: pd.DataFrame,
    output_path: str,
    vocab_size: int = VOCAB_SIZE,
):
    sg_to_symbols = load_wyckoff_symbols()
    wyckoff_mapping = wyckoff_symbol_to_index(sg_to_symbols)

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
        except Exception as e:
            print(f"Error processing {material_id}: {e}")

    # Save processed data as a torch file
    torch.save(processed_data, output_path)
    print(f"Saved processed data to {output_path}")


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

    df = pd.read_csv(args.input_csv)
    process_dataframe(df, args.wyckoff_symbols, args.output, args.vocab_size)
