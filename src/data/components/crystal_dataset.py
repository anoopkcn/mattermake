import torch
from torch.utils.data import Dataset


class CrystalDataset(Dataset):
    """Dataset for crystal structures used in diffusion models."""

    def __init__(
        self,
        file_path,
        max_atoms=80,
        include_properties=True,
        transform=None,
    ):
        """
        Initialize the crystal dataset.

        Args:
            file_path: Path to the preprocessed data file (.pt or .npz)
            max_atoms: Maximum number of atoms to include per structure
            include_properties: Whether to load property data for conditional generation
            transform: Optional transform to apply to the data
        """
        self.max_atoms = max_atoms
        self.include_properties = include_properties
        self.transform = transform

        # Load the data
        try:
            if file_path.endswith('.pt'):
                # Load with weights_only=False since we need to load all Python objects
                self.data = torch.load(file_path, weights_only=False)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")

            # Check that required keys exist
            required_keys = ["atom_types", "positions", "lattice"]
            for key in required_keys:
                if key not in self.data:
                    raise KeyError(f"Required key '{key}' not found in dataset")

            # Convert any non-tensor data to tensors
            for key in required_keys:
                if not isinstance(self.data[key][0], torch.Tensor):
                    dtype = torch.long if key == "atom_types" else torch.float32
                    self.data[key] = [torch.tensor(item, dtype=dtype) for item in self.data[key]]

            print(f"Loaded crystal dataset with {len(self)} structures")

        except Exception as e:
            raise RuntimeError(f"Failed to load dataset from {file_path}: {str(e)}")

    def __len__(self):
        """Return the number of crystal structures in the dataset."""
        return len(self.data["positions"])

    def __getitem__(self, idx):
        """
        Get a crystal structure by index.

        Returns:
            dict: Dictionary containing:
                - crystal: Dictionary with atom_types, positions, lattice
                - condition: Optional property data for conditioning
        """
        # Get basic crystal data
        positions = self.data["positions"][idx]
        atom_types = self.data["atom_types"][idx]
        lattice = self.data["lattice"][idx]

        # Ensure we have tensors
        if not isinstance(positions, torch.Tensor):
            positions = torch.tensor(positions, dtype=torch.float32)
        if not isinstance(atom_types, torch.Tensor):
            atom_types = torch.tensor(atom_types, dtype=torch.long)
        if not isinstance(lattice, torch.Tensor):
            lattice = torch.tensor(lattice, dtype=torch.float32)

        # Handle structures that might have more atoms than our maximum
        if len(positions) > self.max_atoms:
            # Randomly select max_atoms
            indices = torch.randperm(len(positions))[:self.max_atoms]
            positions = positions[indices]
            atom_types = atom_types[indices]
        elif len(positions) < self.max_atoms:
            # Pad with zeros - we'll create a mask to ignore these later
            pad_size = self.max_atoms - len(positions)
            position_padding = torch.zeros((pad_size, 3), dtype=torch.float32, device=positions.device)
            positions = torch.cat([positions, position_padding], dim=0)

            atom_padding = torch.zeros(pad_size, dtype=torch.long, device=atom_types.device)
            atom_types = torch.cat([atom_types, atom_padding], dim=0)

        # Create atom mask (1 for real atoms, 0 for padding)
        atom_mask = torch.ones(self.max_atoms, dtype=torch.bool, device=positions.device)
        num_atoms = len(self.data["positions"][idx])
        if isinstance(num_atoms, torch.Tensor):
            num_atoms = num_atoms.item()
        if num_atoms < self.max_atoms:
            atom_mask[num_atoms:] = False

        # Get condition data if available and requested
        condition = None
        if self.include_properties and "properties" in self.data:
            condition = self.data["properties"][idx]
            if not isinstance(condition, torch.Tensor):
                condition = torch.tensor(condition, dtype=torch.float32)

        # Create the output dictionary
        sample = {
            "crystal": {
                "atom_types": atom_types,
                "positions": positions,
                "lattice": lattice,
                "atom_mask": atom_mask
            }
        }

        if condition is not None:
            sample["condition"] = condition

        # Apply transforms if specified
        if self.transform:
            sample = self.transform(sample)

        return sample
