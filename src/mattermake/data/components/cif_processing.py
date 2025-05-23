from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from typing import List, Tuple, Optional

from mattermake.utils import RankedLogger
from mattermake.data.components.wyckoff_interface import wyckoff_interface

log = RankedLogger(__name__, rank_zero_only=True)


def parse_cif_file(cif_file: str) -> Structure:
    """
    Parse a CIF file into pymatgen structure
    """
    try:
        log.debug(f"Parsing CIF file: {cif_file}")
        with open(cif_file, "r") as f:
            return Structure.from_str(f.read(), fmt="cif")
    except FileNotFoundError:
        log.error(f"File not found: {cif_file}")
        raise FileNotFoundError(f"File not found: {cif_file}")
    except Exception as e:
        log.error(f"Error parsing CIF file {cif_file}: {str(e)}")
        raise


def parse_cif_string(cif_string: str) -> Structure:
    """
    Parses a CIF string into a pymatgen Structure object.
    """
    log.debug("Parsing CIF string")
    try:
        return Structure.from_str(cif_string, fmt="cif")
    except Exception as e:
        log.error(f"Error parsing CIF string: {str(e)}")
        raise


def get_composition_vector(structure: Structure, vocab_size: int = 100) -> List[int]:
    """
    Returns a fixed-length vector of element counts (by atomic number).
    """
    from pymatgen.core.periodic_table import Element

    log.debug(f"Creating composition vector with vocab_size={vocab_size}")
    comp_vec = [0] * vocab_size
    for el, amt in structure.composition.get_el_amt_dict().items():
        Z = Element(el).Z
        # Use Z-1 for 0-based indexing, ensure Z is within bounds
        if 0 < Z <= vocab_size:
            comp_vec[Z - 1] = int(amt)
        else:
            log.warning(f"Element {el} (Z={Z}) is outside vocab_size range, ignoring")
    return comp_vec


def get_spacegroup_number(structure: Structure) -> int:
    """
    Returns the spacegroup number (1-230).
    """
    log.debug("Determining space group number")
    sga = SpacegroupAnalyzer(structure, symprec=0.01)
    sg_num = sga.get_space_group_number()
    log.debug(f"Found space group number: {sg_num}")
    return sg_num


def get_lattice_parameters(structure: Structure) -> List[float]:
    """
    Returns [a, b, c, alpha, beta, gamma] as floats.
    """
    log.debug("Extracting lattice parameters")
    latt = structure.lattice
    return [latt.a, latt.b, latt.c, latt.alpha, latt.beta, latt.gamma]


def get_lattice_matrix(structure: Structure) -> List[float]:
    """
    Returns the 3x3 lattice matrix, flattened.
    """
    log.debug("Extracting lattice matrix")
    # Flatten the 3x3 matrix without numpy
    matrix = structure.lattice.matrix
    return [element for row in matrix for element in row]


def get_wyckoff_symbols_per_atom(structure: Structure) -> List[Optional[str]]:
    """
    Returns a list of Wyckoff symbols, one for each atom in the structure (order matches structure.sites).
    Some positions may not have Wyckoff symbols assigned (None).
    """
    log.debug("Calculating Wyckoff symbols per atom")
    sga = SpacegroupAnalyzer(structure, symprec=0.01)
    symm_struct = sga.get_symmetrized_structure()
    wyckoff_symbols = symm_struct.wyckoff_symbols
    equivalent_sites = symm_struct.equivalent_sites

    # Build a mapping from site index to wyckoff symbol
    site_to_wyckoff: List[Optional[str]] = [None] * len(structure.sites)
    for wyckoff_symbol, site_group in zip(wyckoff_symbols, equivalent_sites):
        for site in site_group:
            idx = structure.sites.index(site)
            site_to_wyckoff[idx] = wyckoff_symbol
    return site_to_wyckoff


def get_asymmetric_unit_atoms(
    structure: Structure,
) -> List[Tuple[int, int, List[float]]]:
    """
    Returns a list of (atomic_number, wyckoff_index, free_coords) for each atom in the asymmetric unit.
    Wyckoff indices are consistent across all space groups using the wyckoff package interface.
    """
    log.debug("Extracting asymmetric unit atoms using wyckoff interface")

    sga = SpacegroupAnalyzer(structure, symprec=0.01)
    symm_struct = sga.get_symmetrized_structure()
    sg_num = sga.get_space_group_number()

    atom_data = []
    for site_group, wyckoff_symbol in zip(
        symm_struct.equivalent_sites, symm_struct.wyckoff_symbols
    ):
        atom = site_group[0]  # Get representative atom from the equivalent site group
        atomic_number = atom.specie.Z

        # Only use the letter part (strip multiplicity if present)
        wyckoff_letter = wyckoff_symbol[-1]

        # Use the wyckoff interface to get a consistent Wyckoff index
        wyckoff_index = wyckoff_interface.wyckoff_to_index(sg_num, wyckoff_letter)

        coords = [float(c) for c in atom.frac_coords]
        atom_data.append((atomic_number, wyckoff_index, coords))
    return atom_data


if __name__ == "__main__":
    import argparse

    log.info("Running CIF processing module as script")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cif_string",
        type=str,
        help="CIF string",
    )
    parser.add_argument("--cif_file", type=str, help="CIF file")
    args = parser.parse_args()

    if args.cif_file:
        log.info(f"Processing CIF file: {args.cif_file}")
        structure = parse_cif_file(args.cif_file)
    elif args.cif_string:
        log.info("Processing CIF string")
        structure = parse_cif_string(args.cif_string)
    else:
        log.error("Must provide either --cif_file or --cif_string")
        raise ValueError("Must provide either --cif_file or --cif_string")

    composition_vector = get_composition_vector(structure, vocab_size=100)
    space_group_number = get_spacegroup_number(structure)
    lattice_parameters = get_lattice_parameters(structure)
    atom_seq = get_asymmetric_unit_atoms(structure)

    print(
        f"composition vector={composition_vector}\nspace_group_number={space_group_number},\nlattice_parameters={lattice_parameters}"
    )

    atom_types = [item[0] for item in atom_seq]
    atom_wyckoffs = [item[1] for item in atom_seq]
    atom_coords = [item[2] for item in atom_seq]

    print(
        f"atom_types = {atom_types}\natom_wycoffs = {atom_wyckoffs}\natom_coords = {atom_coords}"
    )
