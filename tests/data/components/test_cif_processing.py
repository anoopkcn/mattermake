import pytest
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
import os

from mattermake.data.components.cif_processing import (
    parse_cif_string,
    parse_cif_file,
    get_composition_vector,
    get_spacegroup_number,
    get_lattice_parameters,
    get_lattice_matrix,
    get_wyckoff_symbols_per_atom,
    get_asymmetric_unit_atoms,
)

from mattermake.data.components.wyckoff_utils import (
    load_wyckoff_symbols,
    wyckoff_symbol_to_index,
)

# Sample CIF string for testing (NaCl, SG 221)
SAMPLE_CIF_STRING = """
data_NaCl
_symmetry_space_group_name_H-M   'F m -3 m'
_cell_length_a   5.6402
_cell_length_b   5.6402
_cell_length_c   5.6402
_cell_angle_alpha   90
_cell_angle_beta    90
_cell_angle_gamma   90
_symmetry_equiv_pos_site_id   1
_symmetry_equiv_pos_as_xyz   'x,y,z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Na Na 0.00000 0.00000 0.00000 1.0
Cl Cl 0.50000 0.50000 0.50000 1.0
"""


# Fixture for the parsed structure
@pytest.fixture(scope="module")
def nacl_structure() -> Structure:
    """Parses the sample CIF string into a pymatgen Structure."""
    return parse_cif_string(SAMPLE_CIF_STRING)


# Fixture for Wyckoff data needed by get_asymmetric_unit_atoms
@pytest.fixture(scope="module")
def wyckoff_data():
    """Loads Wyckoff symbol data required for some tests."""
    # Minimal check to see if the JSON exists, similar to test_wyckoff_utils
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate up and over to the data location
    data_dir = os.path.join(script_dir, "..", "..", "mattermake", "data", "components")
    json_path = os.path.join(data_dir, "wyckoff_symbols.json")
    if not os.path.exists(json_path):
        pytest.skip(
            f"Wyckoff symbols JSON file not found at expected location: {json_path}"
        )

    sg_to_symbols = (
        load_wyckoff_symbols()
    )  # Assumes load_wyckoff_symbols finds the file correctly
    mapping = wyckoff_symbol_to_index(sg_to_symbols)
    return sg_to_symbols, mapping


# --- Test Functions ---


def test_parse_cif_string(nacl_structure):
    assert isinstance(nacl_structure, Structure)
    assert len(nacl_structure.sites) > 0  # Check that sites were loaded


def test_parse_cif_file():
    # Test FileNotFoundError
    with pytest.raises(FileNotFoundError):
        parse_cif_file("non_existent_file.cif")
    # Note: Testing successful parsing would require creating a temporary file.


def test_get_composition_vector(nacl_structure):
    vocab_size = 100
    comp_vec = get_composition_vector(nacl_structure, vocab_size=vocab_size)
    assert isinstance(comp_vec, list)
    assert len(comp_vec) == vocab_size
    assert comp_vec[Element("Na").Z - 1] == 1
    assert comp_vec[Element("Cl").Z - 1] == 1
    assert comp_vec[Element("H").Z - 1] == 0
    assert comp_vec[Element("O").Z - 1] == 0
    assert sum(comp_vec) == 2


def test_get_spacegroup_number(nacl_structure):
    sg_num = get_spacegroup_number(nacl_structure)
    assert isinstance(sg_num, int)
    assert sg_num == 221  # Expected for Fm-3m


def test_get_lattice_parameters(nacl_structure):
    params = get_lattice_parameters(nacl_structure)
    assert isinstance(params, list)
    assert len(params) == 6
    expected_params = [5.6402, 5.6402, 5.6402, 90.0, 90.0, 90.0]
    # Use approx for floating point comparisons
    assert params == pytest.approx(expected_params)


def test_get_lattice_matrix(nacl_structure):
    matrix_flat = get_lattice_matrix(nacl_structure)
    assert isinstance(matrix_flat, list)
    assert len(matrix_flat) == 9
    a = 5.6402
    # Expected for cubic: [a, 0, 0, 0, a, 0, 0, 0, a]
    expected_matrix_flat = [a, 0.0, 0.0, 0.0, a, 0.0, 0.0, 0.0, a]
    assert matrix_flat == pytest.approx(expected_matrix_flat)


def test_get_wyckoff_symbols_per_atom(nacl_structure):
    symbols = get_wyckoff_symbols_per_atom(nacl_structure)
    assert isinstance(symbols, list)
    assert len(symbols) == len(
        nacl_structure.sites
    )  # Should match number of atoms in structure
    na_count = 0
    cl_count = 0
    expected_na_symbol = "1a"
    expected_cl_symbol = "1b"
    for site, symbol in zip(nacl_structure.sites, symbols):
        if site.specie == Element("Na"):
            assert symbol == expected_na_symbol
            na_count += 1
        elif site.specie == Element("Cl"):
            assert symbol == expected_cl_symbol
            cl_count += 1
    assert na_count == 1
    assert cl_count == 1


def test_get_asymmetric_unit_atoms(nacl_structure, wyckoff_data):
    sg_to_symbols, wyckoff_mapping = wyckoff_data
    atom_data = get_asymmetric_unit_atoms(
        nacl_structure, sg_to_symbols, wyckoff_mapping
    )

    assert isinstance(atom_data, list)
    # NaCl asymmetric unit contains 1 Na and 1 Cl
    assert len(atom_data) == 2

    sg_num = 221
    expected_wyckoff_idx_na = wyckoff_mapping.get((sg_num, "1a"), -1)
    expected_wyckoff_idx_cl = wyckoff_mapping.get((sg_num, "1b"), -1)

    # Sort atom_data by atomic number to ensure consistent order for comparison
    atom_data.sort(key=lambda x: x[0])

    # Check Na atom data (Z=11)
    na_atom = atom_data[0]
    assert na_atom[0] == Element("Na").Z
    assert na_atom[1] == expected_wyckoff_idx_na
    assert na_atom[2] == pytest.approx([0.0, 0.0, 0.0])  # Na is at origin

    # Check Cl atom data (Z=17)
    cl_atom = atom_data[1]
    assert cl_atom[0] == Element("Cl").Z
    assert cl_atom[1] == expected_wyckoff_idx_cl
    assert cl_atom[2] == pytest.approx([0.5, 0.5, 0.5])  # Cl is at (0.5, 0.5, 0.5)

    # Ensure indices were found
    assert expected_wyckoff_idx_na != -1
    assert expected_wyckoff_idx_cl != -1
