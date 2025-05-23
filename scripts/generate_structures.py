"""
Crystal structure generation script for MatterMake.

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
import rootutils

from mattermake.models.modular_hierarchical_crystal_transformer_module import (
    ModularHierarchicalCrystalTransformer,
)
from mattermake.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

class CrystalStructureGenerator:
    """Generator class for creating crystal structures using trained MHCT model."""

    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: str = "auto"
    ):
        """
        Initialize the generator with a trained model.

        Args:
            checkpoint_path: Path to the trained model checkpoint
            config_path: Optional path to model config (if not in checkpoint)
            device: Device to run inference on ("auto", "cpu", "cuda")
        """
        self.checkpoint_path = checkpoint_path
        self.device = self._setup_device(device)
        self.model = self._load_model(config_path)

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

    def _load_model(self, config_path: Optional[str] = None) -> ModularHierarchicalCrystalTransformer:
        """Load the trained model from checkpoint."""
        log.info(f"Loading model from checkpoint: {self.checkpoint_path}")

        try:
            # Load model from checkpoint
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
            raise

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
            composition: Tensor representing target composition
            num_structures: Number of structures to generate
            max_atoms: Maximum number of atoms per structure
            spacegroup: Optional fixed space group (if None, model predicts)
            temperature: Sampling temperature (lower = more deterministic)
            sg_sampling_mode: Space group sampling mode
            lattice_sampling_mode: Lattice parameter sampling mode
            atom_discrete_sampling_mode: Atom type sampling mode
            coord_sampling_mode: Coordinate sampling mode

        Returns:
            List of generated crystal structures
        """
        log.info(f"Generating {num_structures} crystal structures")
        log.info(f"Parameters: max_atoms={max_atoms}, temperature={temperature}")

        generated_structures = []

        with torch.no_grad():
            for i in range(num_structures):
                try:
                    # Move composition to device
                    comp_tensor = composition.to(self.device)
                    sg_tensor = spacegroup.to(self.device) if spacegroup is not None else None

                    # Generate structure
                    result = self.model.generate(
                        composition=comp_tensor,
                        spacegroup=sg_tensor,
                        max_atoms=max_atoms,
                        sg_sampling_mode=sg_sampling_mode,
                        lattice_sampling_mode=lattice_sampling_mode,
                        atom_discrete_sampling_mode=atom_discrete_sampling_mode,
                        coord_sampling_mode=coord_sampling_mode,
                        temperature=temperature,
                    )

                    # Move results back to CPU for storage
                    result_cpu = self._move_to_cpu(result)
                    generated_structures.append(result_cpu)

                    log.info(f"Generated structure {i+1}/{num_structures}")

                except Exception as e:
                    log.warning(f"Failed to generate structure {i+1}: {e}")
                    continue

        return generated_structures

    def _move_to_cpu(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Move tensor data to CPU and convert to numpy/lists for serialization."""
        result = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.cpu().numpy().tolist()
            elif isinstance(value, dict):
                result[key] = self._move_to_cpu(value)
            else:
                result[key] = value
        return result


def create_composition_tensor(elements: List[str], counts: List[int], vocab_size: int = 100) -> torch.Tensor:
    """
    Create a composition tensor from element symbols and counts.

    Args:
        elements: List of element symbols (e.g., ['Si', 'O'])
        counts: List of atom counts (e.g., [1, 2])
        vocab_size: Size of element vocabulary

    Returns:
        Composition tensor suitable for model input
    """
    # This is a simplified example - you'll need to adapt based on your actual data format
    # and element vocabulary mapping

    composition = torch.zeros(1, vocab_size)  # Batch size 1

    # Example mapping (you'll need the actual element-to-index mapping from your training)
    element_to_idx = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
        'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
        'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
        # Add more elements as needed...
    }

    for element, count in zip(elements, counts):
        if element in element_to_idx:
            idx = element_to_idx[element]
            composition[0, idx] = count
        else:
            log.warning(f"Unknown element: {element}")

    return composition


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Generate crystal structures using MatterMake")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--elements",
        type=str,
        nargs="+",
        default=["Si", "O"],
        help="List of elements in composition (e.g., Si O)"
    )
    parser.add_argument(
        "--counts",
        type=int,
        nargs="+",
        default=[1, 2],
        help="List of atom counts (e.g., 1 2 for SiO2)"
    )
    parser.add_argument(
        "--num_structures",
        type=int,
        default=5,
        help="Number of structures to generate"
    )
    parser.add_argument(
        "--max_atoms",
        type=int,
        default=50,
        help="Maximum number of atoms per structure"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="generated_structures.json",
        help="Output file for generated structures"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for inference"
    )

    args = parser.parse_args()

    # Validate inputs
    if len(args.elements) != len(args.counts):
        raise ValueError("Number of elements must match number of counts")

    # Initialize generator
    generator = CrystalStructureGenerator(
        checkpoint_path=args.checkpoint,
        device=args.device
    )

    # Create composition tensor
    composition = create_composition_tensor(args.elements, args.counts)

    log.info(f"Generating structures for composition: {dict(zip(args.elements, args.counts))}")

    # Generate structures
    structures = generator.generate_structures(
        composition=composition,
        num_structures=args.num_structures,
        max_atoms=args.max_atoms,
        temperature=args.temperature,
    )

    # Save results
    output_data = {
        "input_composition": {
            "elements": args.elements,
            "counts": args.counts
        },
        "generation_parameters": {
            "num_structures": args.num_structures,
            "max_atoms": args.max_atoms,
            "temperature": args.temperature,
        },
        "generated_structures": structures
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    log.info(f"Generated {len(structures)} structures and saved to {args.output}")


if __name__ == "__main__":
    main()
