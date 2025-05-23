#!/usr/bin/env python3
"""
Wyckoff Data Cleaning Utility

This script cleans existing datasets by filtering out or correcting invalid Wyckoff indices.
It can handle various types of Wyckoff data issues and provides detailed reporting.

Usage:
    python clean_wyckoff_data.py --input dataset.pt --output cleaned_dataset.pt --filter
    python clean_wyckoff_data.py --input dataset.pt --output cleaned_dataset.pt --correct
    python clean_wyckoff_data.py --input_dir data/ --output_dir cleaned_data/ --filter --recursive
    python clean_wyckoff_data.py --input dataset.pt --analyze_only --report_file wyckoff_report.txt
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
from tqdm import tqdm
import shutil
from collections import defaultdict, Counter

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mattermake.data.components.wyckoff_interface import wyckoff_interface
from mattermake.utils import RankedLogger

# Setup logging
logging.basicConfig(level=logging.INFO)
log = RankedLogger(__name__, rank_zero_only=True)


class WyckoffDataCleaner:
    """Cleans datasets by filtering or correcting invalid Wyckoff indices."""
    
    def __init__(
        self,
        cleaning_mode: str = "filter",  # "filter", "correct", or "analyze"
        strict_validation: bool = True,
        backup_original: bool = True,
        verbose: bool = True
    ):
        """
        Initialize the Wyckoff data cleaner.
        
        Args:
            cleaning_mode: How to handle invalid indices ("filter", "correct", "analyze")
            strict_validation: Whether to use strict space group validation
            backup_original: Whether to create backups of original files
            verbose: Whether to provide detailed output
        """
        self.cleaning_mode = cleaning_mode
        self.strict_validation = strict_validation
        self.backup_original = backup_original
        self.verbose = verbose
        
        # Initialize Wyckoff interface
        self.wyckoff_vocab_size = wyckoff_interface.get_vocab_size()
        log.info(f"Initialized Wyckoff cleaner with vocab size: {self.wyckoff_vocab_size}")
        log.info(f"Cleaning mode: {self.cleaning_mode}")
        log.info(f"Strict validation: {self.strict_validation}")
        
        # Statistics
        self.stats = {
            "total_structures": 0,
            "valid_structures": 0,
            "invalid_structures": 0,
            "filtered_structures": 0,
            "corrected_structures": 0,
            "error_structures": 0,
            "wyckoff_issues": defaultdict(int),
            "spacegroup_distribution": defaultdict(int),
            "correction_log": []
        }
    
    def clean_dataset_file(self, input_file: Path, output_file: Optional[Path] = None) -> Dict[str, Any]:
        """Clean a single dataset file."""
        log.info(f"Cleaning dataset file: {input_file}")
        
        if not input_file.exists():
            raise FileNotFoundError(f"Input file does not exist: {input_file}")
        
        # Create backup if requested
        if self.backup_original and output_file and output_file != input_file:
            backup_file = input_file.with_suffix(".backup" + input_file.suffix)
            shutil.copy2(input_file, backup_file)
            log.info(f"Created backup: {backup_file}")
        
        # Load dataset
        try:
            structures = torch.load(input_file, map_location='cpu')
            self.stats["total_structures"] = len(structures)
            log.info(f"Loaded {len(structures)} structures")
        except Exception as e:
            log.error(f"Error loading dataset: {e}")
            raise
        
        # Process structures
        cleaned_structures = []
        for i, structure in enumerate(tqdm(structures, desc="Processing structures")):
            result = self._process_structure(structure, i)
            
            if result["action"] == "keep":
                cleaned_structures.append(result["structure"])
                self.stats["valid_structures"] += 1
            elif result["action"] == "correct":
                cleaned_structures.append(result["structure"])
                self.stats["corrected_structures"] += 1
            elif result["action"] == "filter":
                self.stats["filtered_structures"] += 1
            else:  # error
                self.stats["error_structures"] += 1
                if self.cleaning_mode != "filter":
                    cleaned_structures.append(structure)  # Keep original if not filtering
        
        # Save cleaned dataset if not in analyze mode
        if self.cleaning_mode != "analyze" and output_file:
            self._save_cleaned_dataset(cleaned_structures, output_file)
        
        return self.stats
    
    def clean_dataset_directory(
        self, 
        input_dir: Path, 
        output_dir: Optional[Path] = None,
        recursive: bool = False,
        pattern: str = "*.pt"
    ) -> Dict[str, Any]:
        """Clean all dataset files in a directory."""
        log.info(f"Cleaning dataset directory: {input_dir}")
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
        
        # Find dataset files
        if recursive:
            dataset_files = list(input_dir.rglob(pattern))
        else:
            dataset_files = list(input_dir.glob(pattern))
        
        if not dataset_files:
            log.warning(f"No files matching pattern '{pattern}' found in {input_dir}")
            return self.stats
        
        log.info(f"Found {len(dataset_files)} dataset files to process")
        
        # Process each file
        for dataset_file in dataset_files:
            try:
                # Determine output file path
                if output_dir:
                    relative_path = dataset_file.relative_to(input_dir)
                    output_file = output_dir / relative_path
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                else:
                    output_file = dataset_file  # Overwrite original
                
                log.info(f"Processing: {dataset_file}")
                self.clean_dataset_file(dataset_file, output_file)
                
            except Exception as e:
                log.error(f"Error processing {dataset_file}: {e}")
                self.stats["error_structures"] += 1
        
        return self.stats
    
    def _process_structure(self, structure: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Process a single structure and determine action to take."""
        material_id = structure.get("material_id", f"structure_{index}")
        
        try:
            # Extract Wyckoff data
            wyckoff_indices = self._extract_wyckoff_indices(structure)
            spacegroup = self._extract_spacegroup(structure)
            
            if wyckoff_indices is None or spacegroup is None:
                self.stats["wyckoff_issues"]["missing_data"] += 1
                return {"action": "error", "structure": structure}
            
            # Update spacegroup distribution
            self.stats["spacegroup_distribution"][spacegroup] += 1
            
            # Validate Wyckoff indices
            validation_result = self._validate_wyckoff_indices(wyckoff_indices, spacegroup, material_id)
            
            if validation_result["is_valid"]:
                return {"action": "keep", "structure": structure}
            
            # Handle invalid Wyckoff indices based on cleaning mode
            if self.cleaning_mode == "filter":
                return {"action": "filter", "structure": structure}
            elif self.cleaning_mode == "correct":
                corrected_structure = self._correct_wyckoff_indices(
                    structure, validation_result, material_id
                )
                if corrected_structure:
                    return {"action": "correct", "structure": corrected_structure}
                else:
                    return {"action": "filter", "structure": structure}
            else:  # analyze mode
                return {"action": "keep", "structure": structure}
                
        except Exception as e:
            log.error(f"Error processing structure {material_id}: {e}")
            self.stats["wyckoff_issues"]["processing_error"] += 1
            return {"action": "error", "structure": structure}
    
    def _extract_wyckoff_indices(self, structure: Dict[str, Any]) -> Optional[List[int]]:
        """Extract Wyckoff indices from structure."""
        if "wyckoff" not in structure:
            return None
        
        wyckoff_data = structure["wyckoff"]
        
        # Convert to list if tensor
        if isinstance(wyckoff_data, torch.Tensor):
            wyckoff_indices = wyckoff_data.tolist()
        elif isinstance(wyckoff_data, list):
            wyckoff_indices = wyckoff_data
        else:
            return None
        
        return wyckoff_indices
    
    def _extract_spacegroup(self, structure: Dict[str, Any]) -> Optional[int]:
        """Extract spacegroup number from structure."""
        if "spacegroup" not in structure:
            return None
        
        sg_data = structure["spacegroup"]
        
        if isinstance(sg_data, torch.Tensor):
            if sg_data.numel() == 1:
                return int(sg_data.item())
            else:
                return None
        elif isinstance(sg_data, (int, float)):
            return int(sg_data)
        else:
            return None
    
    def _validate_wyckoff_indices(
        self, 
        wyckoff_indices: List[int], 
        spacegroup: int, 
        material_id: str
    ) -> Dict[str, Any]:
        """Validate Wyckoff indices and return detailed results."""
        result = {
            "is_valid": True,
            "issues": [],
            "invalid_indices": [],
            "correctable_indices": {}
        }
        
        for i, idx in enumerate(wyckoff_indices):
            issue_type = None
            corrected_idx = None
            
            # Check for special tokens (these are always valid)
            if idx in [-2, -1, 0]:
                continue
            
            # Check bounds
            if idx < -2 or idx >= self.wyckoff_vocab_size:
                issue_type = "out_of_bounds"
                corrected_idx = 0  # Use padding token
                self.stats["wyckoff_issues"]["out_of_bounds"] += 1
            else:
                # Check space group consistency
                try:
                    decoded_sg, decoded_letter = wyckoff_interface.index_to_wyckoff(idx)
                    
                    if decoded_sg != spacegroup and decoded_sg != 0:  # 0 is padding
                        if self.strict_validation:
                            issue_type = "spacegroup_mismatch"
                            # Try to find valid Wyckoff for this space group
                            corrected_idx = self._find_valid_wyckoff_for_sg(spacegroup)
                            self.stats["wyckoff_issues"]["spacegroup_mismatch"] += 1
                        else:
                            # Log warning but consider valid
                            if self.verbose:
                                log.warning(f"Space group mismatch for {material_id}: idx {idx} -> SG {decoded_sg}, expected {spacegroup}")
                            self.stats["wyckoff_issues"]["spacegroup_warning"] += 1
                            
                except Exception as e:
                    issue_type = "decode_error"
                    corrected_idx = 0
                    self.stats["wyckoff_issues"]["decode_error"] += 1
                    if self.verbose:
                        log.error(f"Error decoding Wyckoff index {idx} for {material_id}: {e}")
            
            if issue_type:
                result["is_valid"] = False
                result["issues"].append({
                    "position": i,
                    "index": idx,
                    "type": issue_type,
                    "corrected": corrected_idx
                })
                result["invalid_indices"].append(i)
                if corrected_idx is not None:
                    result["correctable_indices"][i] = corrected_idx
        
        return result
    
    def _find_valid_wyckoff_for_sg(self, spacegroup: int) -> Optional[int]:
        """Find a valid Wyckoff position for the given space group."""
        try:
            valid_letters = wyckoff_interface.get_valid_wyckoff_letters(spacegroup)
            if valid_letters:
                # Use the first valid letter (typically 'a')
                first_letter = valid_letters[0]
                return wyckoff_interface.wyckoff_to_index(spacegroup, first_letter)
        except Exception as e:
            if self.verbose:
                log.error(f"Error finding valid Wyckoff for SG {spacegroup}: {e}")
        return None
    
    def _correct_wyckoff_indices(
        self, 
        structure: Dict[str, Any], 
        validation_result: Dict[str, Any], 
        material_id: str
    ) -> Optional[Dict[str, Any]]:
        """Correct invalid Wyckoff indices in a structure."""
        if not validation_result["correctable_indices"]:
            return None
        
        # Create a copy of the structure
        corrected_structure = structure.copy()
        wyckoff_indices = self._extract_wyckoff_indices(structure)
        
        if wyckoff_indices is None:
            return None
        
        # Apply corrections
        corrected_indices = wyckoff_indices.copy()
        corrections_made = []
        
        for pos, corrected_idx in validation_result["correctable_indices"].items():
            old_idx = corrected_indices[pos]
            corrected_indices[pos] = corrected_idx
            corrections_made.append({
                "position": pos,
                "old_index": old_idx,
                "new_index": corrected_idx
            })
        
        # Update structure with corrected indices
        if isinstance(structure["wyckoff"], torch.Tensor):
            corrected_structure["wyckoff"] = torch.tensor(corrected_indices, dtype=structure["wyckoff"].dtype)
        else:
            corrected_structure["wyckoff"] = corrected_indices
        
        # Log corrections
        correction_entry = {
            "material_id": material_id,
            "corrections": corrections_made
        }
        self.stats["correction_log"].append(correction_entry)
        
        if self.verbose:
            log.info(f"Corrected {len(corrections_made)} Wyckoff indices for {material_id}")
        
        return corrected_structure
    
    def _save_cleaned_dataset(self, structures: List[Dict[str, Any]], output_file: Path) -> None:
        """Save cleaned dataset to file."""
        if not structures:
            log.warning("No structures to save")
            return
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save dataset
        torch.save(structures, output_file)
        log.info(f"Saved {len(structures)} cleaned structures to {output_file}")
    
    def generate_report(self, report_file: Optional[Path] = None) -> str:
        """Generate a detailed cleaning report."""
        report_lines = [
            "=== Wyckoff Data Cleaning Report ===",
            f"Total structures processed: {self.stats['total_structures']}",
            f"Valid structures: {self.stats['valid_structures']}",
            f"Invalid structures: {self.stats['invalid_structures']}",
            f"Corrected structures: {self.stats['corrected_structures']}",
            f"Filtered structures: {self.stats['filtered_structures']}",
            f"Error structures: {self.stats['error_structures']}",
            "",
            "Wyckoff Issues Breakdown:",
        ]
        
        for issue_type, count in self.stats["wyckoff_issues"].items():
            report_lines.append(f"  {issue_type}: {count}")
        
        if self.stats["spacegroup_distribution"]:
            report_lines.extend([
                "",
                "Space Group Distribution (top 10):",
            ])
            top_sgs = sorted(
                self.stats["spacegroup_distribution"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            for sg, count in top_sgs:
                report_lines.append(f"  SG {sg}: {count} structures")
        
        if self.stats["correction_log"]:
            report_lines.extend([
                "",
                f"Correction Details ({len(self.stats['correction_log'])} structures corrected):"
            ])
            
            for entry in self.stats["correction_log"][:10]:  # Show first 10
                material_id = entry["material_id"]
                num_corrections = len(entry["corrections"])
                report_lines.append(f"  {material_id}: {num_corrections} corrections")
                
                for correction in entry["corrections"]:
                    pos = correction["position"]
                    old_idx = correction["old_index"]
                    new_idx = correction["new_index"]
                    report_lines.append(f"    Position {pos}: {old_idx} -> {new_idx}")
            
            if len(self.stats["correction_log"]) > 10:
                remaining = len(self.stats["correction_log"]) - 10
                report_lines.append(f"  ... and {remaining} more corrected structures")
        
        # Calculate statistics
        if self.stats["total_structures"] > 0:
            valid_rate = (self.stats["valid_structures"] / self.stats["total_structures"]) * 100
            report_lines.extend([
                "",
                f"Success rate: {valid_rate:.1f}%"
            ])
            
            if self.stats["corrected_structures"] > 0:
                correction_rate = (self.stats["corrected_structures"] / self.stats["total_structures"]) * 100
                report_lines.append(f"Correction rate: {correction_rate:.1f}%")
            
            if self.stats["filtered_structures"] > 0:
                filter_rate = (self.stats["filtered_structures"] / self.stats["total_structures"]) * 100
                report_lines.append(f"Filter rate: {filter_rate:.1f}%")
        
        report_text = "\n".join(report_lines)
        
        # Save report to file if specified
        if report_file:
            report_file.parent.mkdir(parents=True, exist_ok=True)
            with open(report_file, 'w') as f:
                f.write(report_text)
            log.info(f"Report saved to {report_file}")
        
        return report_text
    
    def print_summary(self) -> None:
        """Print a summary of cleaning results."""
        log.info("=== Wyckoff Cleaning Summary ===")
        log.info(f"Mode: {self.cleaning_mode}")
        log.info(f"Total structures: {self.stats['total_structures']}")
        log.info(f"Valid: {self.stats['valid_structures']}")
        log.info(f"Corrected: {self.stats['corrected_structures']}")
        log.info(f"Filtered: {self.stats['filtered_structures']}")
        log.info(f"Errors: {self.stats['error_structures']}")
        
        if self.stats['total_structures'] > 0:
            retention_rate = ((self.stats['valid_structures'] + self.stats['corrected_structures']) / self.stats['total_structures']) * 100
            log.info(f"Data retention rate: {retention_rate:.1f}%")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Clean Wyckoff data in datasets")
    
    # Input/output options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=Path, help="Input dataset file (.pt)")
    input_group.add_argument("--input_dir", type=Path, help="Input directory containing dataset files")
    
    parser.add_argument("--output", type=Path, help="Output dataset file (.pt)")
    parser.add_argument("--output_dir", type=Path, help="Output directory for cleaned datasets")
    
    # Cleaning mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--filter", action="store_true", help="Filter out structures with invalid Wyckoff indices")
    mode_group.add_argument("--correct", action="store_true", help="Correct invalid Wyckoff indices where possible")
    mode_group.add_argument("--analyze_only", action="store_true", help="Only analyze data, don't modify")
    
    # Processing options
    parser.add_argument("--recursive", action="store_true", help="Process directories recursively")
    parser.add_argument("--pattern", type=str, default="*.pt", help="File pattern for dataset files")
    parser.add_argument("--lenient", action="store_true", help="Use lenient validation (allow space group mismatches)")
    parser.add_argument("--no_backup", action="store_true", help="Don't create backup files")
    
    # Output options
    parser.add_argument("--report_file", type=Path, help="Path to save detailed report")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    # Determine cleaning mode
    if args.filter:
        cleaning_mode = "filter"
    elif args.correct:
        cleaning_mode = "correct"
    elif args.analyze_only:
        cleaning_mode = "analyze"
    else:
        cleaning_mode = "filter"  # Default
    
    # Validate arguments
    if cleaning_mode != "analyze" and not args.output and not args.output_dir:
        parser.error("Output path required unless using --analyze_only")
    
    if args.input_dir and args.output and not args.output_dir:
        parser.error("--output_dir required when using --input_dir")
    
    # Create cleaner
    cleaner = WyckoffDataCleaner(
        cleaning_mode=cleaning_mode,
        strict_validation=not args.lenient,
        backup_original=not args.no_backup,
        verbose=not args.quiet
    )
    
    try:
        # Process datasets
        if args.input:
            cleaner.clean_dataset_file(args.input, args.output)
        elif args.input_dir:
            cleaner.clean_dataset_directory(
                args.input_dir, 
                args.output_dir, 
                args.recursive, 
                args.pattern
            )
        
        # Generate and display report
        report = cleaner.generate_report(args.report_file)
        
        if not args.quiet:
            print("\n" + report)
        
        cleaner.print_summary()
        
    except Exception as e:
        log.error(f"Error during cleaning: {e}")
        sys.exit(1)
    
    log.info("Wyckoff data cleaning completed successfully!")


if __name__ == "__main__":
    main()