import ast
from typing import Dict, List, Tuple
import json
import os

from mattermake.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def load_wyckoff_symbols() -> Dict[int, List[str]]:
    """
    Loads wyckoff symbols from the json file
    Returns a mapping: {spacegroup_number: [wyckoff_symbol1, wyckoff_symbol2, ...]}
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "wyckoff_symbols.json")
    log.info(f"Loading Wyckoff symbols from {json_path}")

    try:
        with open(json_path, "r") as f:
            WYCKOFF_SYMBOLS = json.load(f)

        sg_to_symbols = {}
        for entry in WYCKOFF_SYMBOLS:
            sg = int(entry["sg"])
            symbols = ast.literal_eval(entry["symbols"])
            sg_to_symbols[sg] = symbols

        log.debug(f"Loaded symbols for {len(sg_to_symbols)} space groups")
        return sg_to_symbols

    except FileNotFoundError:
        log.error(f"Wyckoff symbols file not found: {json_path}")
        raise
    except json.JSONDecodeError:
        log.error(f"Error decoding JSON from {json_path}")
        raise
    except Exception as e:
        log.error(f"Error loading Wyckoff symbols: {str(e)}")
        raise


def wyckoff_symbol_to_index(
    sg_to_symbols: Dict[int, List[str]],
) -> Dict[Tuple[int, str], int]:
    """
    Returns a mapping from (spacegroup_number, wyckoff_symbol) -> index
    """
    log.debug("Creating Wyckoff symbol to index mapping")
    mapping = {}
    for sg, symbols in sg_to_symbols.items():
        for idx, symbol in enumerate(symbols, start=1):
            mapping[(sg, symbol)] = idx
    log.debug(f"Created mapping with {len(mapping)} entries")
    return mapping


def get_wyckoff_index(
    spacegroup: int, wyckoff_symbol: str, mapping: Dict[Tuple[int, str], int]
) -> int:
    """
    Returns the integer index for a given (spacegroup, wyckoff_symbol)
    """
    key = (spacegroup, wyckoff_symbol)
    index = mapping.get(key, -1)  # -1 if not found
    if index == -1:
        log.warning(
            f"Wyckoff symbol '{wyckoff_symbol}' not found for space group {spacegroup}"
        )
    return index


def get_wyckoff_symbols_for_sg(
    spacegroup: int, sg_to_symbols: Dict[int, List[str]]
) -> List[str]:
    """
    Returns the list of Wyckoff symbols for a given spacegroup.
    """
    symbols = sg_to_symbols.get(spacegroup, [])
    if not symbols:
        log.warning(f"No Wyckoff symbols found for space group {spacegroup}")
    return symbols


if __name__ == "__main__":
    log.info("Running Wyckoff utilities module as script")

    sg_to_symbols = load_wyckoff_symbols()
    mapping = wyckoff_symbol_to_index(sg_to_symbols)

    # Example usage
    spacegroup = 225
    wyckoff_symbol = "4b"
    index = get_wyckoff_index(spacegroup, wyckoff_symbol, mapping)
    log.info(f"Index for ({spacegroup}, {wyckoff_symbol}): {index}")

    symbols = get_wyckoff_symbols_for_sg(spacegroup, sg_to_symbols)
    log.info(f"Wyckoff symbols for spacegroup {spacegroup}: {symbols}")

    # Additional example usage
    spacegroup = 194
    wyckoff_symbol = "12k"
    index = get_wyckoff_index(spacegroup, wyckoff_symbol, mapping)
    log.info(f"Index for ({spacegroup}, {wyckoff_symbol}): {index}")

    symbols = get_wyckoff_symbols_for_sg(spacegroup, sg_to_symbols)
    log.info(f"Wyckoff symbols for spacegroup {spacegroup}: {symbols}")
