import torch
from pymatgen.core import Structure, Lattice

def crystal_dict_to_structure(crystal_dict, element_map=None):
    """
    Convert a crystal dictionary to a pymatgen Structure.

    Args:
        crystal_dict: Dictionary with atom_types, positions, lattice
        element_map: Dictionary mapping indices to element symbols

    Returns:
        pymatgen.core.Structure
    """
    # Default element map if none provided
    if element_map is None:
        element_map = {
            i+1: element for i, element in enumerate(
                ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
                 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
                 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
                 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
                 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
                 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
                 'Pa', 'U', 'Np', 'Pu', 'Am'])
        }

    # Convert tensors to numpy arrays if needed
    atom_types = crystal_dict['atom_types']
    positions = crystal_dict['positions']
    lattice = crystal_dict['lattice']

    if torch.is_tensor(atom_types):
        atom_types = atom_types.cpu().numpy()
    if torch.is_tensor(positions):
        positions = positions.cpu().numpy()
    if torch.is_tensor(lattice):
        lattice = lattice.cpu().numpy()

    # Get valid atoms (non-zero types)
    valid_idx = atom_types > 0
    valid_types = atom_types[valid_idx]
    valid_positions = positions[valid_idx]

    # Convert atom types to element symbols
    elements = [element_map.get(int(t), 'X') for t in valid_types]

    # Create lattice from parameters
    a, b, c, alpha, beta, gamma = lattice
    lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)

    # Create structure
    structure = Structure(lattice, elements, valid_positions, coords_are_cartesian=False)
    return structure

def structure_to_crystal_dict(structure, max_atoms=80, num_atom_types=95, element_map=None):
    """
    Convert a pymatgen Structure to a crystal dictionary.

    Args:
        structure: pymatgen.core.Structure
        max_atoms: Maximum number of atoms to include
        num_atom_types: Number of atom types in the model
        element_map: Dictionary mapping element symbols to indices

    Returns:
        Dictionary with atom_types, positions, lattice
    """
    # Default element map if none provided (inverse of the above map)
    if element_map is None:
        elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                   'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
                   'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                   'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
                   'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
                   'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
                   'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
                   'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                   'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
                   'Pa', 'U', 'Np', 'Pu', 'Am']
        element_map = {element: i+1 for i, element in enumerate(elements)}

    # Get lattice parameters
    lattice = structure.lattice
    lattice_params = [lattice.a, lattice.b, lattice.c,
                      lattice.alpha, lattice.beta, lattice.gamma]

    # Get positions and atom types
    positions = []
    atom_types = []

    for site in structure.sites:
        positions.append(site.frac_coords)
        element = site.species_string
        atom_type = element_map.get(element, 0)  # Default to 0 if element not found
        atom_types.append(atom_type)

    # Limit to max_atoms
    if len(atom_types) > max_atoms:
        atom_types = atom_types[:max_atoms]
        positions = positions[:max_atoms]

    # Pad if necessary
    if len(atom_types) < max_atoms:
        pad_size = max_atoms - len(atom_types)
        atom_types.extend([0] * pad_size)
        positions.extend([[0, 0, 0]] * pad_size)

    # Create atom mask
    atom_mask = torch.zeros(max_atoms, dtype=torch.bool)
    atom_mask[:len(structure.sites)] = True

    # Create crystal dictionary
    crystal_dict = {
        'atom_types': torch.tensor(atom_types, dtype=torch.long),
        'positions': torch.tensor(positions, dtype=torch.float),
        'lattice': torch.tensor(lattice_params, dtype=torch.float),
        'atom_mask': atom_mask
    }

    return crystal_dict
