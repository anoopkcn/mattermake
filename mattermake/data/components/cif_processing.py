from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from typing import List, Tuple, Dict
import warnings


def parse_cif_file(cif_file: str) -> Structure:
    """
    Parse a CIF file into pymatgen structure
    """
    try:
        with open(cif_file, "r") as f:
            return Structure.from_str(f.read(), fmt="cif")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {cif_file}")


def parse_cif_string(cif_string: str) -> Structure:
    """
    Parses a CIF string into a pymatgen Structure object.
    """
    return Structure.from_str(cif_string, fmt="cif")


def get_composition_vector(structure: Structure, vocab_size: int = 100) -> List[int]:
    """
    Returns a fixed-length vector of element counts (by atomic number).
    """
    from pymatgen.core.periodic_table import Element

    comp_vec = [0] * vocab_size
    for el, amt in structure.composition.get_el_amt_dict().items():
        Z = Element(el).Z
        # Use Z-1 for 0-based indexing, ensure Z is within bounds
        if 0 < Z <= vocab_size:
            comp_vec[Z - 1] = int(amt)
    return comp_vec


def get_spacegroup_number(structure: Structure) -> int:
    """
    Returns the spacegroup number (1-230).
    """
    sga = SpacegroupAnalyzer(structure, symprec=0.01)
    return sga.get_space_group_number()


def get_lattice_parameters(structure: Structure) -> List[float]:
    """
    Returns [a, b, c, alpha, beta, gamma] as floats.
    """
    latt = structure.lattice
    return [latt.a, latt.b, latt.c, latt.alpha, latt.beta, latt.gamma]


def get_lattice_matrix(structure: Structure) -> List[float]:
    """
    Returns the 3x3 lattice matrix, flattened.
    """
    # Flatten the 3x3 matrix without numpy
    matrix = structure.lattice.matrix
    return [element for row in matrix for element in row]


def get_wyckoff_symbols_per_atom(structure: Structure) -> List[str]:
    """
    Returns a list of Wyckoff symbols, one for each atom in the structure (order matches structure.sites).
    """
    sga = SpacegroupAnalyzer(structure, symprec=0.01)
    symm_struct = sga.get_symmetrized_structure()
    wyckoff_symbols = symm_struct.wyckoff_symbols
    equivalent_sites = symm_struct.equivalent_sites

    # Build a mapping from site index to wyckoff symbol
    site_to_wyckoff = [None] * len(structure.sites)
    for wyckoff_symbol, site_group in zip(wyckoff_symbols, equivalent_sites):
        for site in site_group:
            idx = structure.sites.index(site)
            site_to_wyckoff[idx] = wyckoff_symbol
    return site_to_wyckoff


def get_asymmetric_unit_atoms(
    structure: Structure,
    sg_to_symbols: Dict[int, List[str]],
    wyckoff_mapping: Dict[Tuple[int, str], int],
) -> List[Tuple[int, int, List[float]]]:
    """
    Returns a list of (atomic_number, wyckoff_index, free_coords) for each atom in the asymmetric unit.
    """
    sga = SpacegroupAnalyzer(structure, symprec=0.01)
    symm_struct = sga.get_symmetrized_structure()
    sg_num = sga.get_space_group_number()
    # wyckoff_symbols = sg_to_symbols.get(sg_num, [])

    atom_data = []
    for site_group, wyckoff_symbol in zip(
        symm_struct.equivalent_sites, symm_struct.wyckoff_symbols
    ):
        atom = site_group[0]  # Get representative atom from the equivalent site group
        atomic_number = atom.specie.Z

        key = (sg_num, wyckoff_symbol)
        if key in wyckoff_mapping:
            wyckoff_index = wyckoff_mapping[key]
        else:
            warnings.warn(
                f"Wyckoff symbol '{wyckoff_symbol}' not found for space group {sg_num}. "
                f"Using default index 1 for atom {atom.specie} at {atom.frac_coords}.",
                UserWarning,
            )
            wyckoff_index = 1  # TODO:: DO SOMETHING ABOUT THE DEFAULT

        coords = [float(c) for c in atom.frac_coords]
        atom_data.append((atomic_number, wyckoff_index, coords))
    return atom_data


if __name__ == "__main__":
    import argparse
    from mattermake.data.components.wyckoff_utils import (
        load_wyckoff_symbols,
        wyckoff_symbol_to_index,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cif_string",
        type=str,
        help="CIF string",
    )
    parser.add_argument("--cif_file", type=str, help="CIF file")
    args = parser.parse_args()

    if args.cif_file:
        structure = parse_cif_file(args.cif_file)
    elif args.cif_string:
        structure = parse_cif_string(args.cif_string)
    else:
        raise ValueError("Must provide either --cif_file or --cif_string")

    composition_vector = get_composition_vector(structure, vocab_size=100)
    space_group_number = get_spacegroup_number(structure)
    lattice_parameters = get_lattice_parameters(structure)
    # wyckoff_data = get_wyckoff_symbols_per_atom(structure)
    sg_to_symbols = load_wyckoff_symbols()
    mapping = wyckoff_symbol_to_index(sg_to_symbols)
    atom_seq = get_asymmetric_unit_atoms(structure, sg_to_symbols, mapping)

    print(
        f"composition vector={composition_vector}\nspace_group_number={space_group_number},\nlattice_parameters={lattice_parameters}"
    )

    atom_types = [item[0] for item in atom_seq]
    atom_wyckoffs = [item[1] for item in atom_seq]
    atom_coords = [item[2] for item in atom_seq]

    print(
        f"atom_types = {atom_types}\natom_wycoffs = {atom_wyckoffs}\natom_coords = {atom_coords}"
    )
