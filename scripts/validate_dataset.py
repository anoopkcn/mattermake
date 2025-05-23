#!/usr/bin/env python3
"""
This script validates existing datasets for compatibility with the updated Wyckoff encoder system.
It checks data formats, Wyckoff encodings, tensor shapes, and provides detailed reports on any issues.

Usage:
    python validate_dataset.py --dataset_file /path/to/dataset.pt
    python validate_dataset.py --dataset_dir /path/to/dataset/splits --split_mode
    python validate_dataset.py --dataset_file dataset.pt --fix_issues --backup
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import torch
from tqdm import tqdm
import shutil

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mattermake.data.components.wyckoff_interface import wyckoff_interface
from mattermake.data.hct_dataset import HCTDataset
from mattermake.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class DatasetValidator:
    """Validates datasets for Wyckoff encoder compatibility."""

    def __init__(self, fix_issues: bool = False, verbose: bool = True):
        """
        Initialize the validator.

        Args:
            fix_issues: Whether to attempt fixing minor issues
            verbose: Whether to provide detailed output
        """
        self.fix_issues = fix_issues
        self.verbose = verbose

        # Initialize Wyckoff interface
        self.wyckoff_vocab_size = wyckoff_interface.get_vocab_size()

        # Validation statistics
        self.stats = {
            "total_structures": 0,
            "valid_structures": 0,
            "invalid_structures": 0,
            "fixed_structures": 0,
            "errors": [],
            "warnings": []
        }

    def validate_dataset_file(self, dataset_file: Path) -> Dict[str, Any]:
        """Validate a single dataset file."""
        log.info(f"Validating dataset file: {dataset_file}")

        if not dataset_file.exists():
            error = f"Dataset file does not exist: {dataset_file}"
            log.error(error)
            self.stats["errors"].append(error)
            return self.stats

        try:
            # Load dataset
            structures = torch.load(dataset_file)
            self.stats["total_structures"] = len(structures)

            log.info(f"Loaded {len(structures)} structures from {dataset_file}")

            # Validate each structure
            valid_structures = []
            for i, structure in enumerate(tqdm(structures, desc="Validating structures")):
                is_valid, fixed_structure = self.validate_structure(structure, i)

                if is_valid:
                    if fixed_structure is not None:
                        valid_structures.append(fixed_structure)
                        self.stats["fixed_structures"] += 1
                    else:
                        valid_structures.append(structure)
                    self.stats["valid_structures"] += 1
                else:
                    self.stats["invalid_structures"] += 1
                    if not self.fix_issues:
                        valid_structures.append(structure)  # Keep invalid if not fixing

            # Save fixed dataset if issues were fixed
            if self.fix_issues and self.stats["fixed_structures"] > 0:
                self._save_fixed_dataset(valid_structures, dataset_file)

        except Exception as e:
            error = f"Error loading dataset file {dataset_file}: {str(e)}"
            log.error(error)
            self.stats["errors"].append(error)

        return self.stats

    def validate_dataset_directory(self, dataset_dir: Path) -> Dict[str, Any]:
        """Validate dataset directory with train/val/test splits."""
        log.info(f"Validating dataset directory: {dataset_dir}")

        if not dataset_dir.exists():
            error = f"Dataset directory does not exist: {dataset_dir}"
            log.error(error)
            self.stats["errors"].append(error)
            return self.stats

        # Find dataset files
        split_files = []
        for split_name in ["train", "val", "test"]:
            split_dir = dataset_dir / split_name
            if split_dir.exists():
                # Look for .pt files
                pt_files = list(split_dir.glob("*.pt"))
                if pt_files:
                    split_files.extend(pt_files)
                else:
                    warning = f"No .pt files found in {split_dir}"
                    log.warning(warning)
                    self.stats["warnings"].append(warning)

        if not split_files:
            error = f"No dataset files found in {dataset_dir}"
            log.error(error)
            self.stats["errors"].append(error)
            return self.stats

        # Validate each split file
        for split_file in split_files:
            log.info(f"Validating split file: {split_file}")
            split_stats = self.validate_dataset_file(split_file)

            # Aggregate statistics
            for key in ["total_structures", "valid_structures", "invalid_structures", "fixed_structures"]:
                if key in split_stats:
                    self.stats[key] += split_stats[key]

            self.stats["errors"].extend(split_stats.get("errors", []))
            self.stats["warnings"].extend(split_stats.get("warnings", []))

        return self.stats

    def validate_structure(self, structure: Dict[str, Any], index: int) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Validate a single structure.

        Returns:
            Tuple of (is_valid, fixed_structure_or_None)
        """
        is_valid = True
        fixed_structure = None
        structure_id = structure.get("material_id", f"structure_{index}")

        # Check required keys
        required_keys = ["composition", "spacegroup", "lattice", "atom_types", "wyckoff", "atom_coords"]
        missing_keys = [key for key in required_keys if key not in structure]

        if missing_keys:
            error = f"Structure {structure_id}: Missing required keys: {missing_keys}"
            if self.verbose:
                log.error(error)
            self.stats["errors"].append(error)
            return False, None

        # Validate composition
        comp_valid, comp_fixed = self._validate_composition(structure["composition"], structure_id)
        if not comp_valid:
            is_valid = False
        elif comp_fixed is not None:
            if fixed_structure is None:
                fixed_structure = structure.copy()
            fixed_structure["composition"] = comp_fixed

        # Validate spacegroup
        sg_valid, sg_fixed = self._validate_spacegroup(structure["spacegroup"], structure_id)
        if not sg_valid:
            is_valid = False
        elif sg_fixed is not None:
            if fixed_structure is None:
                fixed_structure = structure.copy()
            fixed_structure["spacegroup"] = sg_fixed

        # Validate lattice
        lat_valid, lat_fixed = self._validate_lattice(structure["lattice"], structure_id)
        if not lat_valid:
            is_valid = False
        elif lat_fixed is not None:
            if fixed_structure is None:
                fixed_structure = structure.copy()
            fixed_structure["lattice"] = lat_fixed

        # Validate atom sequences
        seq_valid, seq_fixed = self._validate_atom_sequences(
            structure["atom_types"],
            structure["wyckoff"],
            structure["atom_coords"],
            structure["spacegroup"],
            structure_id
        )
        if not seq_valid:
            is_valid = False
        elif seq_fixed is not None:
            if fixed_structure is None:
                fixed_structure = structure.copy()
            fixed_structure["atom_types"] = seq_fixed[0]
            fixed_structure["wyckoff"] = seq_fixed[1]
            fixed_structure["atom_coords"] = seq_fixed[2]

        return is_valid, fixed_structure

    def _validate_composition(self, composition, structure_id: str) -> Tuple[bool, Optional[torch.Tensor]]:
        """Validate composition vector."""
        try:
            if isinstance(composition, list):
                composition = torch.tensor(composition, dtype=torch.long)
            elif not isinstance(composition, torch.Tensor):
                error = f"Structure {structure_id}: Composition must be tensor or list"
                if self.verbose:
                    log.error(error)
                self.stats["errors"].append(error)
                return False, None

            if composition.dtype != torch.long:
                if self.fix_issues:
                    composition = composition.to(torch.long)
                    warning = f"Structure {structure_id}: Fixed composition dtype"
                    if self.verbose:
                        log.warning(warning)
                    self.stats["warnings"].append(warning)
                    return True, composition
                else:
                    error = f"Structure {structure_id}: Composition must be torch.long"
                    if self.verbose:
                        log.error(error)
                    self.stats["errors"].append(error)
                    return False, None

            if len(composition.shape) != 1:
                error = f"Structure {structure_id}: Composition must be 1D tensor"
                if self.verbose:
                    log.error(error)
                self.stats["errors"].append(error)
                return False, None

            expected_size = 100  # Default element vocab size
            if composition.size(0) != expected_size:
                error = f"Structure {structure_id}: Composition size {composition.size(0)} != {expected_size}"
                if self.verbose:
                    log.error(error)
                self.stats["errors"].append(error)
                return False, None

            return True, None

        except Exception as e:
            error = f"Structure {structure_id}: Error validating composition: {str(e)}"
            if self.verbose:
                log.error(error)
            self.stats["errors"].append(error)
            return False, None

    def _validate_spacegroup(self, spacegroup, structure_id: str) -> Tuple[bool, Optional[int]]:
        """Validate spacegroup number."""
        try:
            if isinstance(spacegroup, torch.Tensor):
                if spacegroup.numel() == 1:
                    sg_num = int(spacegroup.item())
                else:
                    error = f"Structure {structure_id}: Spacegroup tensor must have single element"
                    if self.verbose:
                        log.error(error)
                    self.stats["errors"].append(error)
                    return False, None
            else:
                sg_num = int(spacegroup)

            if not (1 <= sg_num <= 230):
                error = f"Structure {structure_id}: Invalid spacegroup number {sg_num} (must be 1-230)"
                if self.verbose:
                    log.error(error)
                self.stats["errors"].append(error)
                return False, None

            return True, None

        except (ValueError, TypeError) as e:
            error = f"Structure {structure_id}: Error validating spacegroup: {str(e)}"
            if self.verbose:
                log.error(error)
            self.stats["errors"].append(error)
            return False, None

    def _validate_lattice(self, lattice, structure_id: str) -> Tuple[bool, Optional[torch.Tensor]]:
        """Validate lattice parameters."""
        try:
            if isinstance(lattice, list):
                lattice = torch.tensor(lattice, dtype=torch.float)
            elif not isinstance(lattice, torch.Tensor):
                error = f"Structure {structure_id}: Lattice must be tensor or list"
                if self.verbose:
                    log.error(error)
                self.stats["errors"].append(error)
                return False, None

            if lattice.dtype != torch.float:
                if self.fix_issues:
                    lattice = lattice.to(torch.float)
                    warning = f"Structure {structure_id}: Fixed lattice dtype"
                    if self.verbose:
                        log.warning(warning)
                    self.stats["warnings"].append(warning)
                    return True, lattice
                else:
                    error = f"Structure {structure_id}: Lattice must be torch.float"
                    if self.verbose:
                        log.error(error)
                    self.stats["errors"].append(error)
                    return False, None

            if lattice.shape != (6,):
                error = f"Structure {structure_id}: Lattice must have shape (6,), got {lattice.shape}"
                if self.verbose:
                    log.error(error)
                self.stats["errors"].append(error)
                return False, None

            # Check for reasonable lattice parameter values
            a, b, c, alpha, beta, gamma = lattice
            if any(x <= 0 for x in [a, b, c]):
                error = f"Structure {structure_id}: Lattice lengths must be positive"
                if self.verbose:
                    log.error(error)
                self.stats["errors"].append(error)
                return False, None

            if any(not (0 < angle < 180) for angle in [alpha, beta, gamma]):
                error = f"Structure {structure_id}: Lattice angles must be between 0 and 180 degrees"
                if self.verbose:
                    log.error(error)
                self.stats["errors"].append(error)
                return False, None

            return True, None

        except Exception as e:
            error = f"Structure {structure_id}: Error validating lattice: {str(e)}"
            if self.verbose:
                log.error(error)
            self.stats["errors"].append(error)
            return False, None

    def _validate_atom_sequences(self, atom_types, wyckoff, atom_coords, spacegroup, structure_id: str) -> Tuple[bool, Optional[Tuple]]:
        """Validate atom type, Wyckoff, and coordinate sequences."""
        try:
            # Convert to lists if they're tensors
            if isinstance(atom_types, torch.Tensor):
                atom_types = atom_types.tolist()
            if isinstance(wyckoff, torch.Tensor):
                wyckoff = wyckoff.tolist()
            if isinstance(atom_coords, torch.Tensor):
                atom_coords = atom_coords.tolist()

            # Check sequence lengths match
            seq_len = len(atom_types)
            if len(wyckoff) != seq_len:
                error = f"Structure {structure_id}: Wyckoff sequence length {len(wyckoff)} != atom_types length {seq_len}"
                if self.verbose:
                    log.error(error)
                self.stats["errors"].append(error)
                return False, None

            if len(atom_coords) != seq_len:
                error = f"Structure {structure_id}: Coordinates sequence length {len(atom_coords)} != atom_types length {seq_len}"
                if self.verbose:
                    log.error(error)
                self.stats["errors"].append(error)
                return False, None

            # Validate atom types
            fixed_atom_types = None
            for i, at in enumerate(atom_types):
                if not isinstance(at, int):
                    if self.fix_issues:
                        if fixed_atom_types is None:
                            fixed_atom_types = atom_types.copy()
                        fixed_atom_types[i] = int(at)
                    else:
                        error = f"Structure {structure_id}: Atom type at index {i} must be integer"
                        if self.verbose:
                            log.error(error)
                        self.stats["errors"].append(error)
                        return False, None

                # Check for reasonable atomic numbers (including special tokens)
                if at != -1 and at != -2 and not (1 <= at <= 118):
                    error = f"Structure {structure_id}: Invalid atomic number {at} at index {i}"
                    if self.verbose:
                        log.error(error)
                    self.stats["errors"].append(error)
                    return False, None

            # Validate Wyckoff positions
            fixed_wyckoff = None
            sg_num = spacegroup if isinstance(spacegroup, int) else int(spacegroup)

            for i, wp in enumerate(wyckoff):
                if not isinstance(wp, int):
                    if self.fix_issues:
                        if fixed_wyckoff is None:
                            fixed_wyckoff = wyckoff.copy()
                        fixed_wyckoff[i] = int(wp)
                    else:
                        error = f"Structure {structure_id}: Wyckoff index at position {i} must be integer"
                        if self.verbose:
                            log.error(error)
                        self.stats["errors"].append(error)
                        return False, None

                # Check Wyckoff index validity (skip special tokens)
                if wp not in [-1, -2, 0]:  # Allow special tokens and padding
                    if wp < 0 or wp >= self.wyckoff_vocab_size:
                        error = f"Structure {structure_id}: Wyckoff index {wp} out of range [0, {self.wyckoff_vocab_size})"
                        if self.verbose:
                            log.error(error)
                        self.stats["errors"].append(error)
                        return False, None

                    # Validate Wyckoff position for space group
                    sg_decoded, letter_decoded = wyckoff_interface.index_to_wyckoff(wp)
                    if sg_decoded != sg_num:
                        warning = f"Structure {structure_id}: Wyckoff index {wp} maps to SG {sg_decoded}, expected {sg_num}"
                        if self.verbose:
                            log.warning(warning)
                        self.stats["warnings"].append(warning)

            # Validate coordinates
            fixed_coords = None
            for i, coords in enumerate(atom_coords):
                if not isinstance(coords, (list, tuple)):
                    error = f"Structure {structure_id}: Coordinates at index {i} must be list or tuple"
                    if self.verbose:
                        log.error(error)
                    self.stats["errors"].append(error)
                    return False, None

                if len(coords) != 3:
                    error = f"Structure {structure_id}: Coordinates at index {i} must have 3 elements"
                    if self.verbose:
                        log.error(error)
                    self.stats["errors"].append(error)
                    return False, None

                # Check coordinate values are reasonable
                for j, coord in enumerate(coords):
                    if not isinstance(coord, (int, float)):
                        if self.fix_issues:
                            if fixed_coords is None:
                                fixed_coords = [list(c) for c in atom_coords]
                            fixed_coords[i][j] = float(coord)
                        else:
                            error = f"Structure {structure_id}: Coordinate [{i}][{j}] must be numeric"
                            if self.verbose:
                                log.error(error)
                            self.stats["errors"].append(error)
                            return False, None

            # Return fixed sequences if any were fixed
            if any(x is not None for x in [fixed_atom_types, fixed_wyckoff, fixed_coords]):
                return True, (
                    fixed_atom_types or atom_types,
                    fixed_wyckoff or wyckoff,
                    fixed_coords or atom_coords
                )

            return True, None

        except Exception as e:
            error = f"Structure {structure_id}: Error validating sequences: {str(e)}"
            if self.verbose:
                log.error(error)
            self.stats["errors"].append(error)
            return False, None

    def _save_fixed_dataset(self, structures: List[Dict[str, Any]], original_file: Path) -> None:
        """Save fixed dataset to file."""
        if self.fix_issues:
            # Create backup
            backup_file = original_file.with_suffix(".backup" + original_file.suffix)
            shutil.copy2(original_file, backup_file)
            log.info(f"Created backup: {backup_file}")

            # Save fixed dataset
            torch.save(structures, original_file)
            log.info(f"Saved fixed dataset with {len(structures)} structures to {original_file}")

    def test_with_hct_dataset(self, dataset_file: Path) -> bool:
        """Test loading the dataset with HCTDataset class."""
        try:
            log.info(f"Testing dataset loading with HCTDataset...")
            dataset = HCTDataset(
                data_file=dataset_file,
                add_atom_start_token=True,
                add_atom_end_token=True
            )

            log.info(f"Successfully loaded dataset with {len(dataset)} structures")

            # Test loading a sample
            if len(dataset) > 0:
                sample = dataset[0]
                log.info("Sample structure keys and shapes:")
                for key, value in sample.items():
                    if isinstance(value, torch.Tensor):
                        log.info(f"  {key}: {value.shape} {value.dtype}")
                    else:
                        log.info(f"  {key}: {type(value)}")

            return True

        except Exception as e:
            error = f"Error testing with HCTDataset: {str(e)}"
            log.error(error)
            self.stats["errors"].append(error)
            return False

    def print_report(self) -> None:
        """Print validation report."""
        log.info("=== Dataset Validation Report ===")
        log.info(f"Total structures: {self.stats['total_structures']}")
        log.info(f"Valid structures: {self.stats['valid_structures']}")
        log.info(f"Invalid structures: {self.stats['invalid_structures']}")

        if self.stats['fixed_structures'] > 0:
            log.info(f"Fixed structures: {self.stats['fixed_structures']}")

        if self.stats['total_structures'] > 0:
            success_rate = self.stats['valid_structures'] / self.stats['total_structures'] * 100
            log.info(f"Success rate: {success_rate:.1f}%")

        if self.stats['errors']:
            log.info(f"\nErrors ({len(self.stats['errors'])}):")
            for error in self.stats['errors'][:10]:  # Show first 10 errors
                log.info(f"  - {error}")
            if len(self.stats['errors']) > 10:
                log.info(f"  ... and {len(self.stats['errors']) - 10} more errors")

        if self.stats['warnings']:
            log.info(f"\nWarnings ({len(self.stats['warnings'])}):")
            for warning in self.stats['warnings'][:10]:  # Show first 10 warnings
                log.info(f"  - {warning}")
            if len(self.stats['warnings']) > 10:
                log.info(f"  ... and {len(self.stats['warnings']) - 10} more warnings")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Validate dataset for Wyckoff encoder compatibility")

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--dataset_file", type=Path, help="Single dataset file (.pt)")
    input_group.add_argument("--dataset_dir", type=Path, help="Dataset directory with splits")

    # Processing options
    parser.add_argument("--fix_issues", action="store_true", help="Attempt to fix minor issues")
    parser.add_argument("--backup", action="store_true", help="Create backup before fixing (requires --fix_issues)")
    parser.add_argument("--test_loading", action="store_true", help="Test loading with HCTDataset")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    parser.add_argument("--split_mode", action="store_true", help="Validate directory as train/val/test splits")

    args = parser.parse_args()

    if args.backup and not args.fix_issues:
        parser.error("--backup requires --fix_issues")

    # Create validator
    validator = DatasetValidator(fix_issues=args.fix_issues, verbose=args.verbose)

    # Validate dataset
    if args.dataset_file:
        stats = validator.validate_dataset_file(args.dataset_file)

        # Test loading if requested
        if args.test_loading:
            validator.test_with_hct_dataset(args.dataset_file)

    elif args.dataset_dir:
        if args.split_mode:
            stats = validator.validate_dataset_directory(args.dataset_dir)
        else:
            # Validate all .pt files in directory
            pt_files = list(args.dataset_dir.glob("*.pt"))
            if not pt_files:
                log.error(f"No .pt files found in {args.dataset_dir}")
                return

            for pt_file in pt_files:
                log.info(f"Validating {pt_file}")
                validator.validate_dataset_file(pt_file)

    # Print final report
    validator.print_report()

    # Exit with appropriate code
    if validator.stats['invalid_structures'] > 0 or validator.stats['errors']:
        sys.exit(1)
    else:
        log.info("All validations passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
