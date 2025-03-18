import torch


class NormalizePositions:
    """Normalize atomic positions to fractional coordinates."""

    def __call__(self, sample):
        crystal = sample["crystal"]
        positions = crystal["positions"]
        lattice = crystal["lattice"]

        # Convert lattice parameters to matrix
        a, b, c, alpha, beta, gamma = lattice.unbind(-1)
        alpha, beta, gamma = map(lambda x: x * torch.pi / 180, [alpha, beta, gamma])

        # First lattice vector
        v1 = torch.stack([a, torch.zeros_like(a), torch.zeros_like(a)], dim=-1)

        # Second lattice vector
        v2_x = b * torch.cos(gamma)
        v2_y = b * torch.sin(gamma)
        v2 = torch.stack([v2_x, v2_y, torch.zeros_like(b)], dim=-1)

        # Third lattice vector
        v3_x = c * torch.cos(beta)
        v3_y = c * (torch.cos(alpha) - torch.cos(beta) * torch.cos(gamma)) / torch.sin(gamma)
        v3_z = torch.sqrt(c**2 - v3_x**2 - v3_y**2)
        v3 = torch.stack([v3_x, v3_y, v3_z], dim=-1)

        # Create lattice matrix
        lattice_matrix = torch.stack([v1, v2, v3], dim=-2)

        # To convert cartesian coordinates to fractional coordinates:
        # frac_coords = cart_coords @ inv(lattice_matrix)
        inv_lattice = torch.inverse(lattice_matrix)
        fractional_positions = torch.matmul(positions, inv_lattice)

        # Ensure all fractional coordinates are between 0 and 1
        fractional_positions = fractional_positions % 1.0

        # Update the sample
        crystal["positions"] = fractional_positions

        return sample


class OneHotAtomTypes:
    """Convert atom type indices to one-hot vectors."""

    def __init__(self, num_atom_types=95):
        self.num_atom_types = num_atom_types

    def __call__(self, sample):
        crystal = sample["crystal"]
        atom_types = crystal["atom_types"]
        atom_mask = crystal.get("atom_mask", None)

        # Convert to one-hot
        one_hot = torch.zeros((atom_types.size(0), self.num_atom_types),
                               dtype=torch.float32)

        # Only assign one-hot values for actual atoms (not padding)
        if atom_mask is not None:
            valid_indices = torch.nonzero(atom_mask, as_tuple=True)[0]
            valid_atoms = atom_types[valid_indices]
            one_hot[valid_indices, valid_atoms] = 1.0
        else:
            one_hot.scatter_(1, atom_types.unsqueeze(-1), 1.0)

        # Update the sample
        crystal["atom_types_onehot"] = one_hot

        return sample


class RandomRotation:
    """Apply random rotation to positions (data augmentation)."""

    def __call__(self, sample):
        crystal = sample["crystal"]
        positions = crystal["positions"]

        # Create a random rotation matrix (3x3)
        theta = torch.rand(1) * 2 * torch.pi
        phi = torch.rand(1) * 2 * torch.pi
        z = torch.rand(1) * 2 - 1

        # Construct rotation matrix
        r = torch.sqrt(1 - z**2)
        x = r * torch.cos(phi)
        y = r * torch.sin(phi)

        # Normalize the vector
        vec = torch.cat([x, y, z], dim=0)
        vec = vec / torch.norm(vec)

        # Compute rotation matrix using Rodrigues' formula
        cross_product_matrix = torch.tensor([
            [0, -vec[2], vec[1]],
            [vec[2], 0, -vec[0]],
            [-vec[1], vec[0], 0]
        ])

        R = (torch.eye(3) + torch.sin(theta) * cross_product_matrix +
             (1 - torch.cos(theta)) * (cross_product_matrix @ cross_product_matrix))

        # Apply rotation
        rotated_positions = torch.matmul(positions, R.T)
        crystal["positions"] = rotated_positions

        return sample


class ComposeTransforms:
    """Compose multiple transforms."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample
