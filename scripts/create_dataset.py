#!/usr/bin/env python3
"""
This script creates datasets compatible with the updated Wyckoff encoder neural network system.
It processes crystal structure data (CIF files, Materials Project data, etc.) and converts them
to the format expected by the HCTDataset class with proper Wyckoff encoding.

Usage:
    python create_dataset.py --input_dir /path/to/cif_files --output_dir /path/to/output
    python create_dataset.py --mp_api_key YOUR_API_KEY --mp_query '{"elements": ["Li", "O"]}' --output_dir /path/to/output
    python create_dataset.py --cif_file single_structure.cif --output_file structure.pt
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
from tqdm import tqdm
import pandas as pd

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mattermake.data.components.cif_processing import (
    parse_cif_file,
    parse_cif_string,
    get_composition_vector,
    get_spacegroup_number,
    get_lattice_parameters,
    get_asymmetric_unit_atoms,
)
from mattermake.data.components.wyckoff_interface import (
    wyckoff_interface,
)
from mattermake.utils import RankedLogger

# Setup logging
logging.basicConfig(level=logging.INFO)
log = RankedLogger(__name__, rank_zero_only=True)


class DatasetCreator:
    """Creates datasets compatible with the updated Wyckoff encoder system."""

    def __init__(
        self,
        element_vocab_size: int = 100,
        max_structures: Optional[int] = None,
        validate_wyckoff: bool = True,
        filter_invalid_wyckoff: bool = True,
        strict_wyckoff_validation: bool = True,
    ):
        """
        Initialize the dataset creator.

        Args:
            element_vocab_size: Size of element vocabulary (atomic numbers)
            max_structures: Maximum number of structures to process (None for all)
            validate_wyckoff: Whether to validate Wyckoff positions
            filter_invalid_wyckoff: Whether to filter out structures with invalid Wyckoff indices
            strict_wyckoff_validation: Whether to use strict validation (reject space group mismatches)
        """
        self.element_vocab_size = element_vocab_size
        self.max_structures = max_structures
        self.validate_wyckoff = validate_wyckoff
        self.filter_invalid_wyckoff = filter_invalid_wyckoff
        self.strict_wyckoff_validation = strict_wyckoff_validation

        # Initialize Wyckoff interface
        self.wyckoff_vocab_size = wyckoff_interface.get_vocab_size()
        log.info(f"Initialized with Wyckoff vocab size: {self.wyckoff_vocab_size}")
        log.info(f"Wyckoff filtering enabled: {self.filter_invalid_wyckoff}")
        log.info(f"Strict Wyckoff validation: {self.strict_wyckoff_validation}")

        # Statistics
        self.processed_count = 0
        self.error_count = 0
        self.skipped_count = 0
        self.wyckoff_filtered_count = 0
        self.wyckoff_corrected_count = 0

    def process_structure_from_cif_file(
        self, cif_file: Path
    ) -> Optional[Dict[str, Any]]:
        """Process a single CIF file into dataset format."""
        try:
            log.debug(f"Processing CIF file: {cif_file}")
            structure = parse_cif_file(str(cif_file))
            return self.process_structure(structure, material_id=cif_file.stem)
        except Exception as e:
            log.error(f"Error processing CIF file {cif_file}: {str(e)}")
            self.error_count += 1
            return None

    def process_structure_from_cif_string(
        self, cif_string: str, material_id: str = "unknown"
    ) -> Optional[Dict[str, Any]]:
        """Process a CIF string into dataset format."""
        try:
            log.debug(f"Processing CIF string for material: {material_id}")
            structure = parse_cif_string(cif_string)
            return self.process_structure(structure, material_id=material_id)
        except Exception as e:
            log.error(f"Error processing CIF string for {material_id}: {str(e)}")
            self.error_count += 1
            return None

    def process_structure(
        self, structure, material_id: str = "unknown"
    ) -> Optional[Dict[str, Any]]:
        """
        Process a pymatgen Structure object into dataset format.

        Args:
            structure: pymatgen Structure object
            material_id: Identifier for the material

        Returns:
            Dictionary with processed structure data or None if processing failed
        """
        try:
            # Get basic structure information
            composition_vector = get_composition_vector(
                structure, self.element_vocab_size
            )
            spacegroup_number = get_spacegroup_number(structure)
            lattice_params = get_lattice_parameters(structure)

            # Get asymmetric unit atoms with Wyckoff positions
            atom_data = get_asymmetric_unit_atoms(structure)

            if not atom_data:
                log.warning(f"No atoms found in asymmetric unit for {material_id}")
                self.skipped_count += 1
                return None

            # Extract atom types, Wyckoff indices, and coordinates
            atom_types = []
            wyckoff_indices = []
            atom_coords = []

            for atomic_number, wyckoff_idx, coords in atom_data:
                # Validate atomic number
                if atomic_number <= 0 or atomic_number > self.element_vocab_size:
                    log.warning(
                        f"Invalid atomic number {atomic_number} for {material_id}"
                    )
                    continue

                # Comprehensive Wyckoff validation
                if self.validate_wyckoff:
                    wyckoff_valid, corrected_idx = self._validate_wyckoff_index(
                        wyckoff_idx, spacegroup_number, material_id
                    )
                    if not wyckoff_valid:
                        if self.filter_invalid_wyckoff:
                            log.warning(
                                f"Filtering out invalid Wyckoff index {wyckoff_idx} for {material_id}"
                            )
                            continue
                        else:
                            # Use corrected index if available
                            wyckoff_idx = (
                                corrected_idx if corrected_idx is not None else 0
                            )
                            self.wyckoff_corrected_count += 1
                    elif corrected_idx is not None:
                        wyckoff_idx = corrected_idx
                        self.wyckoff_corrected_count += 1

                atom_types.append(atomic_number)
                wyckoff_indices.append(wyckoff_idx)
                atom_coords.append(coords)

            if not atom_types:
                log.warning(f"No valid atoms after validation for {material_id}")
                self.skipped_count += 1
                return None

            # Final Wyckoff validation for the entire structure
            if self.validate_wyckoff and self.filter_invalid_wyckoff:
                if not self._validate_structure_wyckoff_consistency(
                    wyckoff_indices, spacegroup_number, material_id
                ):
                    log.warning(
                        f"Structure {material_id} failed Wyckoff consistency check, filtering out"
                    )
                    self.wyckoff_filtered_count += 1
                    return None

            # Create dataset entry
            data_entry = {
                "composition": composition_vector,
                "spacegroup": spacegroup_number,
                "lattice": lattice_params,
                "atom_types": atom_types,
                "wyckoff": wyckoff_indices,
                "atom_coords": atom_coords,
                "material_id": material_id,
            }

            # Validate the entry
            if self._validate_entry(data_entry):
                self.processed_count += 1
                return data_entry
            else:
                log.warning(f"Failed validation for {material_id}")
                self.skipped_count += 1
                return None

        except Exception as e:
            log.error(f"Error processing structure {material_id}: {str(e)}")
            self.error_count += 1
            return None

    def _validate_entry(self, entry: Dict[str, Any]) -> bool:
        """Validate a dataset entry."""
        try:
            # Check required keys
            required_keys = [
                "composition",
                "spacegroup",
                "lattice",
                "atom_types",
                "wyckoff",
                "atom_coords",
            ]
            for key in required_keys:
                if key not in entry:
                    log.error(f"Missing required key: {key}")
                    return False

            # Validate shapes and types
            if len(entry["composition"]) != self.element_vocab_size:
                log.error(
                    f"Composition vector has wrong size: {len(entry['composition'])} != {self.element_vocab_size}"
                )
                return False

            if not (1 <= entry["spacegroup"] <= 230):
                log.error(f"Invalid spacegroup number: {entry['spacegroup']}")
                return False

            if len(entry["lattice"]) != 6:
                log.error(
                    f"Lattice parameters have wrong size: {len(entry['lattice'])} != 6"
                )
                return False

            # Check that atom sequences have same length
            seq_len = len(entry["atom_types"])
            if len(entry["wyckoff"]) != seq_len:
                log.error(
                    f"Wyckoff sequence length mismatch: {len(entry['wyckoff'])} != {seq_len}"
                )
                return False

            if len(entry["atom_coords"]) != seq_len:
                log.error(
                    f"Coordinates sequence length mismatch: {len(entry['atom_coords'])} != {seq_len}"
                )
                return False

            # Validate coordinate dimensions
            for i, coords in enumerate(entry["atom_coords"]):
                if len(coords) != 3:
                    log.error(
                        f"Coordinates at index {i} have wrong dimension: {len(coords)} != 3"
                    )
                    return False

            # Enhanced Wyckoff validation
            if self.validate_wyckoff:
                if not self._validate_entry_wyckoff_indices(entry):
                    return False

            return True

        except Exception as e:
            log.error(f"Error during validation: {str(e)}")
            return False

    def create_dataset_from_cif_directory(
        self, input_dir: Path, output_file: Path, pattern: str = "*.cif"
    ) -> None:
        """Create dataset from directory of CIF files."""
        # Ensure input_dir and output_file are Path objects
        input_dir = Path(input_dir)
        output_file = Path(output_file)

        log.info(f"Creating dataset from CIF directory: {input_dir}")

        cif_files = list(input_dir.glob(pattern))
        if not cif_files:
            log.error(f"No CIF files found in {input_dir} with pattern {pattern}")
            return

        if self.max_structures:
            cif_files = cif_files[: self.max_structures]

        log.info(f"Found {len(cif_files)} CIF files to process")

        structures = []
        for cif_file in tqdm(cif_files, desc="Processing CIF files"):
            entry = self.process_structure_from_cif_file(cif_file)
            if entry is not None:
                structures.append(entry)

        self._save_dataset(structures, output_file)

    def create_dataset_from_csv(
        self,
        csv_file: Path,
        cif_column: str,
        output_file: Path,
        id_column: Optional[str] = None,
        max_structures: Optional[int] = None,
    ) -> None:
        """Create dataset from CSV file with CIF strings."""
        # Ensure csv_file and output_file are Path objects
        csv_file = Path(csv_file)
        output_file = Path(output_file)

        log.info(f"Creating dataset from CSV file: {csv_file}")

        try:
            # Load CSV file
            df = pd.read_csv(csv_file)
            log.info(f"Loaded CSV with {len(df)} rows")

            # Check if CIF column exists
            if cif_column not in df.columns:
                log.error(
                    f"CIF column '{cif_column}' not found in CSV. Available columns: {list(df.columns)}"
                )
                return

            # Use ID column if specified, otherwise use row index
            if id_column and id_column in df.columns:
                material_ids = df[id_column].astype(str).tolist()
            else:
                material_ids = [f"material_{i}" for i in range(len(df))]

            # Limit structures if specified
            if max_structures and max_structures < len(df):
                df = df.head(max_structures)
                material_ids = material_ids[:max_structures]
                log.info(f"Limited to {max_structures} structures")

            structures = []
            for i, (_, row) in enumerate(
                tqdm(df.iterrows(), desc="Processing CIF strings", total=len(df))
            ):
                try:
                    cif_string = row[cif_column]
                    material_id = material_ids[i]

                    # Skip if CIF string is empty or NaN
                    try:
                        is_null = (
                            pd.isna(cif_string).item()
                            if hasattr(pd.isna(cif_string), "item")
                            else bool(pd.isna(cif_string))
                        )
                    except (AttributeError, ValueError):
                        is_null = cif_string is None or (
                            isinstance(cif_string, str)
                            and cif_string.lower() in ["nan", "null", ""]
                        )

                    cif_str = str(cif_string) if not is_null else ""
                    if not cif_str.strip():
                        log.warning(f"Empty CIF string for {material_id}, skipping")
                        self.skipped_count += 1
                        continue

                    entry = self.process_structure_from_cif_string(cif_str, material_id)
                    if entry is not None:
                        structures.append(entry)

                except Exception as e:
                    try:
                        current_material_id = (
                            material_ids[i] if i < len(material_ids) else f"row_{i}"
                        )
                    except:
                        current_material_id = f"row_{i}"
                    log.error(
                        f"Error processing row {i} ({current_material_id}): {str(e)}"
                    )
                    self.error_count += 1

            self._save_dataset(structures, output_file)

        except Exception as e:
            log.error(f"Error reading CSV file {csv_file}: {str(e)}")
            self.error_count += 1

    def create_dataset_from_structure_list(
        self, structures: List[Any], material_ids: List[str], output_file: Path
    ) -> None:
        """Create dataset from list of pymatgen structures."""
        # Ensure output_file is a Path object
        output_file = Path(output_file)

        log.info(f"Creating dataset from {len(structures)} structures")

        if len(structures) != len(material_ids):
            log.error("Number of structures and material IDs must match")
            return

        dataset_entries = []
        for structure, material_id in tqdm(
            zip(structures, material_ids), desc="Processing structures"
        ):
            entry = self.process_structure(structure, material_id)
            if entry is not None:
                dataset_entries.append(entry)

        self._save_dataset(dataset_entries, output_file)

    def _save_dataset(
        self, structures: List[Dict[str, Any]], output_file: Path
    ) -> None:
        """Save processed structures to file."""
        # Ensure output_file is a Path object
        output_file = Path(output_file)

        if not structures:
            log.error("No valid structures to save")
            return

        # Convert to tensors for compatibility with HCTDataset
        tensor_structures = []
        for entry in structures:
            tensor_entry = {
                "composition": torch.tensor(entry["composition"], dtype=torch.long),
                "spacegroup": entry[
                    "spacegroup"
                ],  # Will be converted to tensor in dataset
                "lattice": torch.tensor(entry["lattice"], dtype=torch.float),
                "atom_types": entry[
                    "atom_types"
                ],  # Will be converted to tensor in dataset
                "wyckoff": entry["wyckoff"],  # Will be converted to tensor in dataset
                "atom_coords": entry[
                    "atom_coords"
                ],  # Will be converted to tensor in dataset
                "material_id": entry["material_id"],
            }
            tensor_structures.append(tensor_entry)

        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save as PyTorch file
        torch.save(tensor_structures, output_file)

        log.info(f"Saved {len(tensor_structures)} structures to {output_file}")
        self._print_statistics()

    def _validate_wyckoff_index(
        self, wyckoff_idx: int, spacegroup_num: int, material_id: str
    ) -> tuple[bool, Optional[int]]:
        """
        Validate a single Wyckoff index.

        Returns:
            Tuple of (is_valid, corrected_index_or_None)
        """
        try:
            # Check for special tokens (start/end/pad)
            if wyckoff_idx in [-2, -1, 0]:
                return True, None

            # Check bounds
            if wyckoff_idx < 0 or wyckoff_idx >= self.wyckoff_vocab_size:
                log.warning(
                    f"Wyckoff index {wyckoff_idx} out of bounds [0, {self.wyckoff_vocab_size}) for {material_id}"
                )
                return False, 0  # Use padding index as fallback

            # Decode Wyckoff index to check space group consistency
            try:
                decoded_sg, decoded_letter = wyckoff_interface.index_to_wyckoff(
                    wyckoff_idx
                )

                if decoded_sg == 0 and decoded_letter == "pad":
                    # This is the padding token, which is valid
                    return True, None

                if decoded_sg != spacegroup_num:
                    warning_msg = f"Wyckoff index {wyckoff_idx} maps to space group {decoded_sg}, expected {spacegroup_num} for {material_id}"
                    log.warning(warning_msg)

                    if self.strict_wyckoff_validation:
                        # Try to find a valid Wyckoff position for this space group
                        corrected_idx = self._find_valid_wyckoff_for_sg(spacegroup_num)
                        if corrected_idx is not None:
                            log.info(
                                f"Corrected Wyckoff index from {wyckoff_idx} to {corrected_idx} for {material_id}"
                            )
                            return True, corrected_idx
                        else:
                            return False, 0
                    else:
                        # Allow but warn
                        return True, None

                return True, None

            except Exception as e:
                log.error(
                    f"Error decoding Wyckoff index {wyckoff_idx} for {material_id}: {e}"
                )
                return False, 0

        except Exception as e:
            log.error(
                f"Error validating Wyckoff index {wyckoff_idx} for {material_id}: {e}"
            )
            return False, 0

    def _find_valid_wyckoff_for_sg(self, spacegroup_num: int) -> Optional[int]:
        """Find a valid Wyckoff position for the given space group."""
        try:
            valid_letters = wyckoff_interface.get_valid_wyckoff_letters(spacegroup_num)
            if valid_letters:
                # Use the first valid letter (typically 'a')
                first_letter = valid_letters[0]
                return wyckoff_interface.wyckoff_to_index(spacegroup_num, first_letter)
        except Exception as e:
            log.error(
                f"Error finding valid Wyckoff for space group {spacegroup_num}: {e}"
            )
        return None

    def _validate_structure_wyckoff_consistency(
        self, wyckoff_indices: List[int], spacegroup_num: int, material_id: str
    ) -> bool:
        """Validate that all Wyckoff indices in a structure are consistent."""
        if not wyckoff_indices:
            return True

        # Check for too many out-of-bounds indices
        invalid_count = 0
        for idx in wyckoff_indices:
            if idx < -2 or idx >= self.wyckoff_vocab_size:
                invalid_count += 1

        # If more than 50% of indices are invalid, reject the structure
        if invalid_count / len(wyckoff_indices) > 0.5:
            log.warning(
                f"Structure {material_id} has {invalid_count}/{len(wyckoff_indices)} invalid Wyckoff indices"
            )
            return False

        return True

    def _validate_entry_wyckoff_indices(self, entry: Dict[str, Any]) -> bool:
        """Validate Wyckoff indices in a dataset entry."""
        try:
            wyckoff_indices = entry["wyckoff"]
            material_id = entry.get("material_id", "unknown")

            # Check each Wyckoff index
            for i, idx in enumerate(wyckoff_indices):
                if not isinstance(idx, int):
                    log.error(
                        f"Wyckoff index at position {i} is not integer: {type(idx)}"
                    )
                    return False

                # Check bounds
                if idx < -2 or idx >= self.wyckoff_vocab_size:
                    log.error(
                        f"Wyckoff index {idx} at position {i} out of bounds for {material_id}"
                    )
                    return False

            return True

        except Exception as e:
            log.error(f"Error validating entry Wyckoff indices: {e}")
            return False

    def _print_statistics(self) -> None:
        """Print processing statistics."""
        total = (
            self.processed_count
            + self.error_count
            + self.skipped_count
            + self.wyckoff_filtered_count
        )
        log.info("=== Processing Statistics ===")
        log.info(f"Total processed: {total}")
        log.info(f"Successfully processed: {self.processed_count}")
        log.info(f"Errors: {self.error_count}")
        log.info(f"Skipped: {self.skipped_count}")
        log.info(f"Wyckoff filtered: {self.wyckoff_filtered_count}")
        log.info(f"Wyckoff corrected: {self.wyckoff_corrected_count}")
        if total > 0:
            log.info(f"Success rate: {self.processed_count / total * 100:.1f}%")
            if self.wyckoff_filtered_count > 0:
                log.info(
                    f"Wyckoff filter rate: {self.wyckoff_filtered_count / total * 100:.1f}%"
                )


def split_dataset(
    input_file: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
) -> None:
    """Split dataset into train/val/test sets."""
    # Ensure input_file and output_dir are Path objects
    input_file = Path(input_file)
    output_dir = Path(output_dir)

    log.info(f"Splitting dataset from {input_file}")

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        log.error("Train, val, and test ratios must sum to 1.0")
        return

    # Load dataset
    structures = torch.load(input_file)
    n_total = len(structures)

    # Shuffle if requested
    if shuffle:
        indices = torch.randperm(n_total).tolist()
        structures = [structures[i] for i in indices]

    # Calculate split sizes
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    # n_test = n_total - n_train - n_val

    # Split the data
    train_data = structures[:n_train]
    val_data = structures[n_train : n_train + n_val]
    test_data = structures[n_train + n_val :]

    # Create output directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"

    for split_dir in [train_dir, val_dir, test_dir]:
        split_dir.mkdir(parents=True, exist_ok=True)

    # Save splits
    torch.save(train_data, train_dir / "data.pt")
    torch.save(val_data, val_dir / "data.pt")
    torch.save(test_data, test_dir / "data.pt")

    log.info("Dataset split complete:")
    log.info(f"  Train: {len(train_data)} samples -> {train_dir / 'data.pt'}")
    log.info(f"  Val: {len(val_data)} samples -> {val_dir / 'data.pt'}")
    log.info(f"  Test: {len(test_data)} samples -> {test_dir / 'data.pt'}")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Create dataset for Wyckoff encoder system"
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_dir", type=Path, help="Directory containing CIF files"
    )
    input_group.add_argument("--cif_file", type=Path, help="Single CIF file to process")
    input_group.add_argument("--csv_file", type=Path, help="CSV file with CIF strings")

    # Output options
    parser.add_argument("--output_file", type=Path, help="Output file path (.pt)")
    parser.add_argument("--output_dir", type=Path, help="Output directory")

    # CSV options
    parser.add_argument(
        "--cif_column", type=str, help="Name of column containing CIF strings"
    )
    parser.add_argument(
        "--id_column",
        type=str,
        help="Name of column containing material IDs (optional)",
    )

    # Processing options
    parser.add_argument(
        "--max_structures", type=int, help="Maximum number of structures to process"
    )
    parser.add_argument(
        "--element_vocab_size", type=int, default=100, help="Element vocabulary size"
    )
    parser.add_argument(
        "--cif_pattern", type=str, default="*.cif", help="Pattern for CIF files"
    )

    # Wyckoff filtering options
    parser.add_argument(
        "--no_wyckoff_filter",
        action="store_true",
        help="Disable Wyckoff index filtering",
    )
    parser.add_argument(
        "--lenient_wyckoff",
        action="store_true",
        help="Use lenient Wyckoff validation (allow space group mismatches)",
    )
    parser.add_argument(
        "--keep_invalid_wyckoff",
        action="store_true",
        help="Keep structures with invalid Wyckoff indices (correct them instead)",
    )

    # Dataset splitting options
    parser.add_argument(
        "--split_dataset", action="store_true", help="Split dataset into train/val/test"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Training set ratio"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.1, help="Validation set ratio"
    )
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test set ratio")

    args = parser.parse_args()

    # Validate arguments
    if args.csv_file and not args.cif_column:
        parser.error("--cif_column required when using --csv_file")

    if not args.output_file and not args.output_dir:
        parser.error("Either --output_file or --output_dir must be specified")

    # Create dataset creator
    creator = DatasetCreator(
        element_vocab_size=args.element_vocab_size,
        max_structures=args.max_structures,
        filter_invalid_wyckoff=not args.no_wyckoff_filter,
        strict_wyckoff_validation=not args.lenient_wyckoff,
    )

    # Determine output file
    if args.output_file:
        output_file = args.output_file
    else:
        output_file = Path(args.output_dir) / "dataset.pt"

    # Process based on input type
    if args.input_dir:
        creator.create_dataset_from_cif_directory(
            args.input_dir, output_file, args.cif_pattern
        )
    elif args.cif_file:
        entry = creator.process_structure_from_cif_file(args.cif_file)
        if entry:
            creator._save_dataset([entry], output_file)
    elif args.csv_file:
        creator.create_dataset_from_csv(
            args.csv_file,
            args.cif_column,
            output_file,
            args.id_column,
            args.max_structures,
        )

    # Split dataset if requested
    if args.split_dataset and args.output_dir:
        split_dataset(
            output_file,
            args.output_dir,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
        )


if __name__ == "__main__":
    main()
