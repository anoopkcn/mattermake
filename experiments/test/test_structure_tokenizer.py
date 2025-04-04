from pymatgen.core import Structure
from mattermake.data.crystal_tokenizer import CrystalTokenizer
import argparse


def test_composition_first_tokenization(file):
    structure = Structure.from_file(file)

    tokenizer = CrystalTokenizer(max_sequence_length=256)

    token_data = tokenizer.tokenize_structure(structure)

    print(f"Structure composition: {token_data.composition}")

    # Print token sequence with segment identifiers

    print("\nToken sequence:")
    for i, (token, segment) in enumerate(
        zip(token_data.sequence, token_data.segment_ids)
    ):
        segment_name = {
            0: "SPECIAL",
            1: "COMPOSITION",
            2: "SPACE_GROUP",
            3: "LATTICE",
            4: "ELEMENT",
            5: "WYCKOFF",
            6: "COORDINATE",
        }.get(segment, "UNKNOWN")

        token_name = tokenizer.idx_to_token.get(token, "UNKNOWN")
        print(f"  {i}: {token_name} (ID: {token}, Segment: {segment_name})")

    return token_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenize a crystal structure from a CIF file."
    )
    parser.add_argument("--file", type=str, help="Path to the CIF file to tokenize")
    args = parser.parse_args()

    test_composition_first_tokenization(args.file)
