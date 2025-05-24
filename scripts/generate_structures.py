"""
Crystal structure generation script for MatterMake v2.

This script loads a trained Modular Hierarchical Crystal Transformer model
and generates new crystal structures from given compositions.

USAGE:
python generate_structures.py \
    --checkpoint models/best_model.ckpt \
    --elements Ba Ti O \
    --counts 1 1 3 \
    --num_structures 5 \
    --max_atoms 20
"""

import argparse
import json
from typing import Any, Dict, List, Optional

import torch
import numpy as np

from mattermake.models.modular_hierarchical_crystal_transformer_module import (
    ModularHierarchicalCrystalTransformer,
)
from mattermake.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class CrystalStructureGenerator:
    """Generator class for creating crystal structures using trained models."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "auto",
    ):
        """
        Initialize the generator with a trained model.

        Args:
            checkpoint_path: Path to the trained model checkpoint
            device: Device to run inference on ("auto", "cpu", "cuda")
        """
        self.checkpoint_path = checkpoint_path
        self.device = self._setup_device(device)
        self.model = self._load_model()

    def _setup_device(self, device: str) -> torch.device:
        """Setup the appropriate device for inference."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)

    def _load_model(self) -> ModularHierarchicalCrystalTransformer:
        """Load the trained model from checkpoint."""
        log.info(f"Loading model from checkpoint: {self.checkpoint_path}")

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

            if 'hyper_parameters' in checkpoint:
                hparams = checkpoint['hyper_parameters']
                log.info("Using hyperparameters from checkpoint")

                # Filter out non-model hyperparameters
                model_hparams = {k: v for k, v in hparams.items()
                               if not k.startswith(('trainer', 'data', 'paths', 'logger', 'callbacks'))}

                # Create model with checkpoint hyperparameters
                model = ModularHierarchicalCrystalTransformer(**model_hparams)
                model.load_state_dict(checkpoint['state_dict'])
            else:
                # Use Lightning's load_from_checkpoint
                model = ModularHierarchicalCrystalTransformer.load_from_checkpoint(
                    self.checkpoint_path,
                    map_location=self.device
                )

            model.eval()
            model.to(self.device)
            log.info("Model loaded successfully")
            return model

        except Exception as e:
            log.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Could not load model from checkpoint: {e}")

    def generate_structures(
        self,
        composition: torch.Tensor,
        num_structures: int = 5,
        max_atoms: int = 50,
        spacegroup: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        sg_sampling_mode: str = "sample",
        lattice_sampling_mode: str = "sample",
        atom_discrete_sampling_mode: str = "sample",
        coord_sampling_mode: str = "sample",
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple crystal structures from a given composition.

        Args:
            composition: Tensor of shape (1, num_elements) representing element counts
            num_structures: Number of structures to generate
            max_atoms: Maximum number of atoms per structure
            spacegroup: Optional space group specification
            temperature: Sampling temperature for randomness control
            sg_sampling_mode: Space group sampling mode ("sample" or "argmax")
            lattice_sampling_mode: Lattice sampling mode ("sample" or "mean")
            atom_discrete_sampling_mode: Atom type sampling mode ("sample" or "argmax")
            coord_sampling_mode: Coordinate sampling mode ("sample" or "mean")

        Returns:
            List of generated structure dictionaries
        """
        structures = []

        log.info(f"Generating {num_structures} structures...")

        for i in range(num_structures):
            log.info(f"Generating structure {i+1}/{num_structures}")

            try:
                with torch.no_grad():
                    result = self.model.generate(
                        composition=composition,
                        spacegroup=spacegroup,
                        max_atoms=max_atoms,
                        temperature=temperature,
                        sg_sampling_mode=sg_sampling_mode,
                        lattice_sampling_mode=lattice_sampling_mode,
                        atom_discrete_sampling_mode=atom_discrete_sampling_mode,
                        coord_sampling_mode=coord_sampling_mode,
                    )

                # Process and clean the generated structure
                structure = self._process_generated_structure(result, i)
                structures.append(structure)

            except Exception as e:
                log.error(f"Failed to generate structure {i+1}: {e}")
                continue

        log.info(f"Successfully generated {len(structures)} structures")
        return structures

    def _process_generated_structure(self, result: Dict[str, Any], structure_id: int) -> Dict[str, Any]:
        """Process and clean a generated structure result."""

        # Extract components
        spacegroup = result.get("spacegroup", torch.tensor([1]))
        lattice_matrix = result.get("lattice_matrix_sampled", torch.eye(3))
        atom_types = result.get("atom_types_sampled", torch.tensor([]))
        coordinates = result.get("coordinates_sampled", torch.tensor([]))
        wyckoff_positions = result.get("wyckoff_sampled", torch.tensor([]))

        # Convert to numpy for easier processing
        if isinstance(spacegroup, torch.Tensor):
            spacegroup = spacegroup.cpu().numpy()
        if isinstance(lattice_matrix, torch.Tensor):
            lattice_matrix = lattice_matrix.cpu().numpy()
        if isinstance(atom_types, torch.Tensor):
            atom_types = atom_types.cpu().numpy()
        if isinstance(coordinates, torch.Tensor):
            coordinates = coordinates.cpu().numpy()
        if isinstance(wyckoff_positions, torch.Tensor):
            wyckoff_positions = wyckoff_positions.cpu().numpy()

        # Clean up the structure
        structure = {
            "structure_id": structure_id,
            "spacegroup": int(spacegroup.item()) if spacegroup.size == 1 else spacegroup.tolist(),
            "lattice_matrix": lattice_matrix.reshape(3, 3).tolist(),
            "atom_types": atom_types.tolist() if atom_types.size > 0 else [],
            "coordinates": coordinates.tolist() if coordinates.size > 0 else [],
            "wyckoff_positions": wyckoff_positions.tolist() if wyckoff_positions.size > 0 else [],
            "num_atoms": len(atom_types) if atom_types.size > 0 else 0,
        }

        return structure

    def save_structures(self, structures: List[Dict[str, Any]], output_path: str):
        """Save generated structures to a JSON file."""
        log.info(f"Saving {len(structures)} structures to {output_path}")

        with open(output_path, 'w') as f:
            json.dump(structures, f, indent=2)

        log.info("Structures saved successfully")


def create_composition_tensor(elements: List[str], counts: List[int], vocab_size: int = 100) -> torch.Tensor:
    """
    Create a composition tensor from element symbols and counts.

    Args:
        elements: List of element symbols (e.g., ['Ba', 'Ti', 'O'])
        counts: List of element counts (e.g., [1, 1, 3])
        vocab_size: Size of element vocabulary

    Returns:
        Composition tensor of shape (1, vocab_size)
    """
    # Simple mapping of common elements to atomic numbers
    element_to_atomic_num = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
        'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
        'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
        'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
        'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
        'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
        'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
        'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
        'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100
    }

    composition = torch.zeros(1, vocab_size, dtype=torch.float32)

    for element, count in zip(elements, counts):
        if element in element_to_atomic_num:
            atomic_num = element_to_atomic_num[element]
            if atomic_num < vocab_size:
                composition[0, atomic_num] = float(count)
            else:
                log.warning(f"Atomic number {atomic_num} for element {element} exceeds vocab_size {vocab_size}")
        else:
            log.warning(f"Unknown element: {element}")

    return composition


def main():
    parser = argparse.ArgumentParser(description="Generate crystal structures using MatterMake")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--elements",
        nargs="+",
        required=True,
        help="Element symbols (e.g., Ba Ti O)"
    )
    parser.add_argument(
        "--counts",
        nargs="+",
        type=int,
        required=True,
        help="Element counts (e.g., 1 1 3)"
    )
    parser.add_argument(
        "--num_structures",
        type=int,
        default=5,
        help="Number of structures to generate (default: 5)"
    )
    parser.add_argument(
        "--max_atoms",
        type=int,
        default=50,
        help="Maximum number of atoms per structure (default: 50)"
    )
    parser.add_argument(
        "--spacegroup",
        type=int,
        default=None,
        help="Optional space group number (1-230)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="generated_structures.json",
        help="Output file path (default: generated_structures.json)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for inference (default: auto)"
    )

    args = parser.parse_args()

    # Validate inputs
    if len(args.elements) != len(args.counts):
        log.error("Number of elements must match number of counts")
        return

    if args.spacegroup is not None and (args.spacegroup < 1 or args.spacegroup > 230):
        log.error("Space group must be between 1 and 230")
        return

    # Create composition tensor
    log.info(f"Creating composition from elements: {args.elements} with counts: {args.counts}")
    composition = create_composition_tensor(args.elements, args.counts)

    # Prepare space group tensor if provided
    spacegroup = None
    if args.spacegroup is not None:
        spacegroup = torch.tensor([[args.spacegroup]], dtype=torch.long)
        log.info(f"Using specified space group: {args.spacegroup}")

    # Initialize generator
    log.info(f"Loading model from: {args.checkpoint}")
    generator = CrystalStructureGenerator(
        checkpoint_path=args.checkpoint,
        device=args.device
    )

    # Generate structures
    structures = generator.generate_structures(
        composition=composition,
        num_structures=args.num_structures,
        max_atoms=args.max_atoms,
        spacegroup=spacegroup,
        temperature=args.temperature
    )

    # Save results
    generator.save_structures(structures, args.output)

    # Print summary
    log.info("Generation complete!")
    log.info(f"Generated {len(structures)} structures")
    log.info(f"Results saved to: {args.output}")

    if structures:
        log.info("Sample structure:")
        sample = structures[0]
        log.info(f"  Space group: {sample['spacegroup']}")
        log.info(f"  Number of atoms: {sample['num_atoms']}")
        log.info(f"  Lattice matrix shape: {np.array(sample['lattice_matrix']).shape}")


if __name__ == "__main__":
    main()
