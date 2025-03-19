import torch
from pymatgen.core import Structure, Lattice, Element


def decode_node_features(node_features):
    """Convert node features back to elements and fractional coordinates."""
    # First 100 features are one-hot encoded elements
    element_probs = node_features[:, :100]
    element_indices = torch.argmax(element_probs, dim=1)

    # Convert indices to element symbols
    elements = [Element.from_Z(idx.item() + 1) for idx in element_indices]

    # Last 3 features are fractional coordinates
    frac_coords = node_features[:, -3:].cpu().numpy()

    return elements, frac_coords


def adjacency_to_edge_index(adjacency, threshold=0.5):
    """Convert adjacency matrix to edge indices."""
    # Apply threshold to get binary adjacency
    binary_adj = (adjacency > threshold).float()

    # Convert to edge index
    edge_index = torch.nonzero(binary_adj, as_tuple=False).t()

    return edge_index


def decode_graph_to_structure(decoded_output, probability_threshold=0.5):
    """Convert a decoded graph from the VAE back to a pymatgen Structure."""
    # Get node features and predicted number of nodes
    node_features = decoded_output["node_features"][0]  # Take first batch item
    num_nodes_logits = decoded_output["num_nodes_logits"][0]
    num_nodes = (
        torch.argmax(num_nodes_logits).item() + 1
    )  # +1 because indices start at 0

    # Trim to predicted number of nodes
    node_features = node_features[:num_nodes]

    # Decode node features to get elements and fractional coordinates
    elements, frac_coords = decode_node_features(node_features)

    # Get cell parameters
    cell_params = decoded_output["cell_params"][0].cpu().numpy()
    a, b, c, alpha, beta, gamma = cell_params

    # Ensure sensible cell parameters
    a, b, c = max(2.0, a), max(2.0, b), max(2.0, c)
    alpha = max(30.0, min(150.0, alpha))
    beta = max(30.0, min(150.0, beta))
    gamma = max(30.0, min(150.0, gamma))

    # Create lattice
    lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)

    # Get adjacency matrix and edge features
    # adjacency = decoded_output["edge_logits"][0, :num_nodes, :num_nodes]
    # edge_index = adjacency_to_edge_index(adjacency, threshold=probability_threshold)

    # Create structure
    structure = Structure(lattice, elements, frac_coords)

    # Get edge features for reconstructing periodicity
    # In a real implementation, we would use the edge feature generator to get periodicity vectors
    # For simplicity, we'll just return the structure without reconstructing edge periodicity

    return structure
