import os
import torch
import argparse
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset
from pymatgen.core import Structure

from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

def convert_to_structure(cif_str):
    """Convert a CIF string to a pymatgen Structure object."""
    try:
        return Structure.from_str(cif_str, fmt="cif")
    except Exception as e:
        log.warning(f"Failed to parse CIF: {str(e)}")
        return None


def structure_to_dict(structure):
    """Convert a pymatgen Structure to a dictionary with needed fields."""
    if structure is None:
        return None

    try:
        # Extract lattice parameters
        lattice = structure.lattice
        lattice_params = [
            lattice.a, lattice.b, lattice.c,
            lattice.alpha, lattice.beta, lattice.gamma
        ]

        # Get atomic positions (fractional coordinates)
        positions = [site.frac_coords for site in structure.sites]

        # Get atomic elements
        elements = [site.species_string for site in structure.sites]

        return {
            "lattice": np.array(lattice_params, dtype=np.float32),
            "positions": np.array(positions, dtype=np.float32),
            "elements": elements,
        }
    except Exception as e:
        log.warning(f"Failed to convert structure to dict: {str(e)}")
        return None


def create_element_mapping(structures):
    """Create a mapping from element symbols to indices."""
    elements = set()
    for structure in structures:
        if structure and "elements" in structure:
            elements.update(structure["elements"])

    # Sort elements
    element_list = sorted(list(elements))

    # Create mapping (reserve 0 for padding or unknown)
    element_map = {elem: idx + 1 for idx, elem in enumerate(element_list)}

    log.info(f"Created element mapping with {len(element_map)} elements")
    return element_map


def convert_elements_to_indices(elements, element_map):
    """Convert element symbols to numerical indices."""
    indices = []
    for element in elements:
        if element in element_map:
            indices.append(element_map[element])
        else:
            # Use 0 for unknown elements
            indices.append(0)

    return np.array(indices, dtype=np.int64)


def prepare_mattext_data(
    output_dir="data/processed",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    max_structures=None,
    batch_size=100,
    random_seed=42
):
    """
    Prepare crystal structure data from MatText dataset for diffusion model training.

    Args:
        output_dir: Directory to save processed data
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        max_structures: Maximum number of structures to process (None for all)
        batch_size: Batch size for processing
        random_seed: Random seed for reproducibility
    """
    os.makedirs(output_dir, exist_ok=True)

    log.info("Loading MatText dataset...")
    dataset = load_dataset("n0w0f/MatText", "pretrain2m", split="train")

    if max_structures is not None and max_structures > 0:
        log.info(f"Limiting to {max_structures} structures")
        dataset = dataset.select(range(min(max_structures, len(dataset))))

    # Only extract the CIF data
    df = dataset.to_pandas()[["material_id", "cif_p1"]]
    log.info(f"Loaded {len(df)} records from dataset")

    # Process structures in batches to conserve memory
    total_batches = (len(df) + batch_size - 1) // batch_size
    all_structures = []

    log.info("Converting CIF strings to structures...")
    for i in tqdm(range(total_batches)):
        batch_df = df.iloc[i * batch_size:(i + 1) * batch_size]

        # Convert CIF strings to structures
        structures = []
        for cif_str in batch_df['cif_p1']:
            structure = convert_to_structure(cif_str)
            structure_dict = structure_to_dict(structure) if structure else None
            structures.append(structure_dict)

        # Filter out None values
        structures = [s for s in structures if s is not None]
        all_structures.extend(structures)

    log.info(f"Successfully converted {len(all_structures)} valid structures")

    # Create element mapping from all structures
    element_map = create_element_mapping(all_structures)

    # Save element mapping
    element_map_path = os.path.join(output_dir, "element_map.pt")
    torch.save(element_map, element_map_path)
    log.info(f"Saved element mapping with {len(element_map)} elements to {element_map_path}")

    # Convert element strings to indices
    for structure in all_structures:
        atom_types = convert_elements_to_indices(
            structure["elements"],
            element_map
        )

        # Replace elements with atom types
        structure["atom_types"] = atom_types
        del structure["elements"]

    # Split into train/val/test
    np.random.seed(random_seed)
    indices = np.random.permutation(len(all_structures))

    train_end = int(len(all_structures) * train_ratio)
    val_end = train_end + int(len(all_structures) * val_ratio)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_structures = [all_structures[i] for i in train_indices]
    val_structures = [all_structures[i] for i in val_indices]
    test_structures = [all_structures[i] for i in test_indices]

    log.info(f"Split data into {len(train_structures)} train, {len(val_structures)} val, {len(test_structures)} test")

    # Process and save each split
    for name, structures in [("train", train_structures), ("val", val_structures), ("test", test_structures)]:
        # Skip empty splits
        if not structures:
            continue

        # Extract data fields
        atom_types = [s["atom_types"] for s in structures]
        positions = [s["positions"] for s in structures]
        lattices = [s["lattice"] for s in structures]

        # Save to file
        output_data = {
            "atom_types": atom_types,
            "positions": positions,
            "lattice": lattices,
        }

        output_path = os.path.join(output_dir, f"{name}.pt")
        torch.save(output_data, output_path)
        log.info(f"Saved {name} data to {output_path}")

    # Save metadata
    metadata = {
        "num_structures": len(all_structures),
        "num_train": len(train_structures),
        "num_val": len(val_structures),
        "num_test": len(test_structures),
        "element_map": element_map,
        "num_elements": len(element_map),
        "has_properties": False,
    }

    metadata_path = os.path.join(output_dir, "metadata.pt")
    torch.save(metadata, metadata_path)
    log.info(f"Saved metadata to {metadata_path}")
    log.info("Data preparation completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare MatText dataset for crystal diffusion")

    parser.add_argument("--output", type=str, default="data/processed",
                        help="Directory to save processed data")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Fraction of data for training")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Fraction of data for validation")
    parser.add_argument("--test-ratio", type=float, default=0.1,
                        help="Fraction of data for testing")
    parser.add_argument("--max-structures", type=int, default=None,
                        help="Maximum number of structures to process (None for all)")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Batch size for processing")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    prepare_mattext_data(
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_structures=args.max_structures,
        batch_size=args.batch_size,
        random_seed=args.seed
    )
