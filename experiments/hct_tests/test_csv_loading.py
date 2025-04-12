import os
import pandas as pd
import tempfile
from mattermake.utils.hct_sequence_utils import process_structure_from_cif_string
from mattermake.data.hct_tokenizer import CrystalTokenizer


def create_test_csv(output_path, num_samples=2):
    """Create a small test CSV with CIF strings"""
    test_cifs = [
        """data_Si
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   5.43095
_cell_length_b   5.43095
_cell_length_c   5.43095
_cell_angle_alpha   90.00000
_cell_angle_beta   90.00000
_cell_angle_gamma   90.00000
_symmetry_Int_Tables_number   1
_chemical_formula_structural   Si
_chemical_formula_sum   'Si8'
_cell_volume   160.2
_cell_formula_units_Z   8
loop_
  _symmetry_equiv_pos_site_id
  _symmetry_equiv_pos_as_xyz
   1  'x, y, z'
loop_
  _atom_site_type_symbol
  _atom_site_label
  _atom_site_symmetry_multiplicity
  _atom_site_fract_x
  _atom_site_fract_y
  _atom_site_fract_z
  _atom_site_occupancy
   Si  Si1  1  0.00000  0.00000  0.00000  1.0
   Si  Si2  1  0.25000  0.25000  0.25000  1.0
   Si  Si3  1  0.00000  0.50000  0.50000  1.0
   Si  Si4  1  0.25000  0.75000  0.75000  1.0
   Si  Si5  1  0.50000  0.00000  0.50000  1.0
   Si  Si6  1  0.75000  0.25000  0.75000  1.0
   Si  Si7  1  0.50000  0.50000  0.00000  1.0
   Si  Si8  1  0.75000  0.75000  0.25000  1.0""",
        """data_NaCl
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   5.64000
_cell_length_b   5.64000
_cell_length_c   5.64000
_cell_angle_alpha   90.00000
_cell_angle_beta   90.00000
_cell_angle_gamma   90.00000
_symmetry_Int_Tables_number   1
_chemical_formula_structural   NaCl
_chemical_formula_sum   'Na4 Cl4'
_cell_volume   179.5
_cell_formula_units_Z   4
loop_
  _symmetry_equiv_pos_site_id
  _symmetry_equiv_pos_as_xyz
   1  'x, y, z'
loop_
  _atom_site_type_symbol
  _atom_site_label
  _atom_site_symmetry_multiplicity
  _atom_site_fract_x
  _atom_site_fract_y
  _atom_site_fract_z
  _atom_site_occupancy
   Na  Na1  1  0.00000  0.00000  0.00000  1.0
   Na  Na2  1  0.00000  0.50000  0.50000  1.0
   Na  Na3  1  0.50000  0.00000  0.50000  1.0
   Na  Na4  1  0.50000  0.50000  0.00000  1.0
   Cl  Cl1  1  0.50000  0.50000  0.50000  1.0
   Cl  Cl2  1  0.50000  0.00000  0.00000  1.0
   Cl  Cl3  1  0.00000  0.50000  0.00000  1.0
   Cl  Cl4  1  0.00000  0.00000  0.50000  1.0""",
    ]

    df = pd.DataFrame(
        {
            "material_id": [f"test_{i}" for i in range(num_samples)],
            "cif": test_cifs[:num_samples],
        }
    )

    df.to_csv(output_path, index=False)
    print(f"Created test CSV at {output_path} with {num_samples} structures")
    return output_path


def test_csv_processing():
    """Test the CSV processing functionality"""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_structures.csv")
        create_test_csv(csv_path)

        tokenizer = CrystalTokenizer(max_sequence_length=256)

        df = pd.read_csv(csv_path)
        first_row = df.iloc[1]

        material_id = first_row["material_id"]
        cif_string = first_row["cif"]

        result = process_structure_from_cif_string(
            material_id, cif_string, tokenizer, standardize=True
        )

        if result:
            print(f"Successfully processed structure {material_id}")
            print(f"Formula: {result['formula']}")
            print(f"Space group: {result['space_group']}")
            print(f"Token sequence length: {len(result['token_data'].sequence)}")
            print(f"Number of atoms: {len(result['structure'])}")

            tokens = result["token_data"].sequence
            token_names = [
                tokenizer.idx_to_token.get(t, f"Unknown-{t}") for t in tokens
            ]
            print(f"Tokens: {token_names}")

            return True
        else:
            print("Failed to process the structure")
            return False


if __name__ == "__main__":
    test_csv_processing()
