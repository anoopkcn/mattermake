import torch
import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.local_env import EconNN


def structure_to_quotient_graph(structure: Structure):
    """
    Convert a pymatgen Structure to a quotient graph representation.

    Args:
        structure: Pymatgen Structure object

    Returns:
        node_features: Tensor of node features [num_nodes, node_feat_dim]
        edge_index: Tensor of edge indices [2, num_edges]
        edge_features: Tensor of edge features including periodicity [num_edges, edge_feat_dim]
        cell_params: Unit cell parameters
    """
    nn_finder = EconNN()

    sites = structure.sites
    num_sites = len(sites)

    atomic_numbers = [site.specie.Z for site in sites]

    node_features = torch.zeros((num_sites, 100), dtype=torch.float)
    for i, z in enumerate(atomic_numbers):
        node_features[i, z - 1] = 1

    frac_coords = np.array([site.frac_coords for site in sites], dtype=np.float32)
    frac_coords = torch.from_numpy(frac_coords)
    node_features = torch.cat([node_features, frac_coords], dim=1)

    edge_index = []
    edge_features = []

    for i, site in enumerate(sites):
        neighbors = nn_finder.get_nn_info(structure, i)
        for neighbor in neighbors:
            j = neighbor["site_index"]
            image = neighbor.get("image", [0, 0, 0])
            edge_index.append([i, j])
            distance = neighbor["weight"]  # ECoN
            edge_features.append([distance] + list(image))

    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    edge_features = torch.tensor(edge_features, dtype=torch.float)

    cell_params = torch.tensor(structure.lattice.parameters, dtype=torch.float)

    return node_features, edge_index, edge_features, cell_params
