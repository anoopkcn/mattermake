from typing import Dict, Tuple, List
import json
import os
import ast
import re

from mattermake.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

# Cache file for the mapping
WYCKOFF_MAPPING_FILE = os.path.join(os.path.dirname(__file__), "wyckoff_mapping.json")


def parse_wyckoff_symbols(symbols_str: str) -> List[str]:
    """
    Parse the Wyckoff symbols string from the JSON file
    
    Args:
        symbols_str: String representation of a list of symbols like "['1a', '2b']"
        
    Returns:
        List of Wyckoff symbols
    """
    try:
        # Use ast.literal_eval for safe evaluation of Python expressions
        return ast.literal_eval(symbols_str)
    except Exception as e:
        log.error(f"Error parsing Wyckoff symbols string: {symbols_str}, {str(e)}")
        # Try a simple regex approach as fallback
        pattern = r"'([^']+)'"
        matches = re.findall(pattern, symbols_str)
        if matches:
            return matches
        return []


def get_wyckoff_letter(symbol: str) -> str:
    """
    Extract the Wyckoff letter from a symbol like '4a' -> 'a'
    
    Args:
        symbol: The Wyckoff symbol (e.g., '1a', '2b')
        
    Returns:
        The Wyckoff letter
    """
    # The letter is always the last character
    return symbol[-1]


def load_wyckoff_data() -> List[Dict]:
    """
    Load the Wyckoff data from the JSON file
    
    Returns:
        List of dictionaries with sg (space group) and symbols (Wyckoff symbols)
    """
    try:
        with open(WYCKOFF_MAPPING_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        log.error(f"Error loading Wyckoff data from {WYCKOFF_MAPPING_FILE}: {str(e)}")
        # Return an empty list as fallback - will generate default mapping
        return []


def create_global_wyckoff_mapping() -> Dict[Tuple[int, str], int]:
    """
    Creates a global mapping of (space_group_number, wyckoff_letter) -> unique_index
    where indices start from 1 (0 is reserved for padding).
    
    Returns:
        Dict[Tuple[int, str], int]: Dictionary mapping (sg_num, wyckoff_letter) to a unique index
    """
    log.info("Creating global Wyckoff position mapping")
    
    mapping = {}
    current_idx = 1  # Start from 1 as 0 is reserved for padding
    
    # Load raw data from the JSON file
    wyckoff_data = load_wyckoff_data()
    
    if not wyckoff_data:
        log.warning("No Wyckoff data loaded. Creating a default mapping.")
        # Create a default mapping with letters a-z for each space group
        letters = [chr(ord('a') + i) for i in range(26)]
        for sg_num in range(1, 231):
            for letter in letters:
                key = (sg_num, letter)
                mapping[key] = current_idx
                current_idx += 1
        return mapping
    
    # First pass: collect all unique letters from all space groups
    all_letters = set()
    for entry in wyckoff_data:
        sg_num = entry.get("sg", 0)
        symbols_str = entry.get("symbols", "[]")
        symbols = parse_wyckoff_symbols(symbols_str)
        
        for symbol in symbols:
            letter = get_wyckoff_letter(symbol)
            all_letters.add(letter)
    
    # Sort letters to ensure consistent ordering
    sorted_letters = sorted(list(all_letters))
    log.info(f"Found {len(sorted_letters)} unique Wyckoff letters")
    
    # Second pass: create the mapping
    for sg_num in range(1, 231):
        # Find the entry for this space group
        sg_entry = next((e for e in wyckoff_data if e.get("sg") == sg_num), None)
        
        if sg_entry:
            symbols_str = sg_entry.get("symbols", "[]")
            symbols = parse_wyckoff_symbols(symbols_str)
            sg_letters = set(get_wyckoff_letter(symbol) for symbol in symbols)
            
            # Add entries for all letters that exist for this space group
            for letter in sorted_letters:
                if letter in sg_letters:
                    key = (sg_num, letter)
                    mapping[key] = current_idx
                    current_idx += 1
        else:
            # If we don't have data for this space group, add all letters
            for letter in sorted_letters:
                key = (sg_num, letter)
                mapping[key] = current_idx
                current_idx += 1
    
    log.info(f"Created global Wyckoff mapping with {len(mapping)} entries")
    return mapping


def get_global_wyckoff_index(sg_num: int, wyckoff_letter: str, mapping: Dict = None) -> int:
    """
    Get the global index for a given space group number and Wyckoff letter.
    
    Args:
        sg_num (int): Space group number (1-230)
        wyckoff_letter (str): Wyckoff letter (a, b, c, ...)
        mapping (Dict, optional): Pre-computed mapping. If None, will load or create mapping.
    
    Returns:
        int: Global Wyckoff index (u22651)
    """
    if mapping is None:
        mapping = load_or_create_wyckoff_mapping()
    
    key = (sg_num, wyckoff_letter)
    if key not in mapping:
        log.warning(f"Unknown Wyckoff position: {key}, returning default index 1")
        return 1
    
    return mapping[key]


def load_or_create_wyckoff_mapping() -> Dict[Tuple[int, str], int]:
    """
    Loads the Wyckoff mapping from cache or creates it.
    
    Returns:
        Dict[Tuple[int, str], int]: The global Wyckoff mapping
    """
    # Create mapping directly from the JSON data
    mapping = create_global_wyckoff_mapping()
    
    # Create a "mapping.cache" file to avoid regenerating the mapping each time
    cache_file = os.path.join(os.path.dirname(__file__), "wyckoff_mapping.cache.json")
    try:
        # Convert tuple keys to strings for JSON serialization
        string_mapping = {str(k): v for k, v in mapping.items()}
        with open(cache_file, 'w') as f:
            json.dump(string_mapping, f, indent=2)
        log.info(f"Saved Wyckoff mapping cache to {cache_file}")
    except Exception as e:
        log.error(f"Error saving Wyckoff mapping cache: {str(e)}")
    
    return mapping


def get_wyckoff_mapping_size() -> int:
    """
    Returns the total number of unique Wyckoff positions across all space groups.
    
    Returns:
        int: Number of unique Wyckoff positions + 1 (for padding)
    """
    mapping = load_or_create_wyckoff_mapping()
    return max(mapping.values()) + 1  # +1 for padding index 0


if __name__ == "__main__":
    """
    Run this as a script to generate and test the Wyckoff mapping
    """
    # Create the mapping
    mapping = load_or_create_wyckoff_mapping()
    print(f"Generated mapping with {len(mapping)} entries")
    
    # Test a few space groups
    test_sg_nums = [1, 2, 3, 4, 47, 123, 221, 230]
    for sg_num in test_sg_nums:
        # Load data from the JSON file
        wyckoff_data = load_wyckoff_data()
        sg_entry = next((e for e in wyckoff_data if e.get("sg") == sg_num), None)
        
        if sg_entry:
            symbols_str = sg_entry.get("symbols", "[]")
            symbols = parse_wyckoff_symbols(symbols_str)
            print(f"Space group {sg_num}: Wyckoff symbols = {symbols}")
            
            # Test mapping for each letter
            for symbol in symbols:
                letter = get_wyckoff_letter(symbol)
                index = get_global_wyckoff_index(sg_num, letter, mapping)
                print(f"  ({sg_num}, {letter}) -> {index}")
        else:
            print(f"No data found for space group {sg_num}")