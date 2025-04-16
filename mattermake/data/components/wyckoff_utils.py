import ast
from typing import Dict, List, Tuple
import json
import os


def load_wyckoff_symbols() -> Dict[int, List[str]]:
    """
    Loads wyckoff symbols from the json file
    Returns a mapping: {spacegroup_number: [wyckoff_symbol1, wyckoff_symbol2, ...]}
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "wyckoff_symbols.json")
    with open(json_path, "r") as f:
        WYCKOFF_SYMBOLS = json.load(f)

    sg_to_symbols = {}
    for entry in WYCKOFF_SYMBOLS:
        sg = int(entry["sg"])
        symbols = ast.literal_eval(entry["symbols"])
        sg_to_symbols[sg] = symbols
    return sg_to_symbols


def wyckoff_symbol_to_index(
    sg_to_symbols: Dict[int, List[str]],
) -> Dict[Tuple[int, str], int]:
    """
    Returns a mapping from (spacegroup_number, wyckoff_symbol) -> index
    """
    mapping = {}
    for sg, symbols in sg_to_symbols.items():
        for idx, symbol in enumerate(symbols, start=1):
            mapping[(sg, symbol)] = idx
    return mapping


def get_wyckoff_index(
    spacegroup: int, wyckoff_symbol: str, mapping: Dict[Tuple[int, str], int]
) -> int:
    """
    Returns the integer index for a given (spacegroup, wyckoff_symbol)
    """
    return mapping.get((spacegroup, wyckoff_symbol), -1)  # -1 if not found


def get_wyckoff_symbols_for_sg(
    spacegroup: int, sg_to_symbols: Dict[int, List[str]]
) -> List[str]:
    """
    Returns the list of Wyckoff symbols for a given spacegroup.
    """
    return sg_to_symbols.get(spacegroup, [])


if __name__ == "__main__":
    sg_to_symbols = load_wyckoff_symbols()
    mapping = wyckoff_symbol_to_index(sg_to_symbols)

    # Example usage
    spacegroup = 225
    wyckoff_symbol = "4b"
    index = get_wyckoff_index(spacegroup, wyckoff_symbol, mapping)
    print(f"Index for ({spacegroup}, {wyckoff_symbol}): {index}")

    symbols = get_wyckoff_symbols_for_sg(spacegroup, sg_to_symbols)
    print(f"Wyckoff symbols for spacegroup {spacegroup}: {symbols}")

    # Additional example usage
    spacegroup = 194
    wyckoff_symbol = "12k"
    index = get_wyckoff_index(spacegroup, wyckoff_symbol, mapping)
    print(f"Index for ({spacegroup}, {wyckoff_symbol}): {index}")

    symbols = get_wyckoff_symbols_for_sg(spacegroup, sg_to_symbols)
    print(f"Wyckoff symbols for spacegroup {spacegroup}: {symbols}")
