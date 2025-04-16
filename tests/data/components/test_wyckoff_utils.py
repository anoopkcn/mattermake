import pytest
import os
from typing import Dict, List, Tuple

# Adjust the import path based on your project structure and how you run pytest
from mattermake.data.components.wyckoff_utils import (
    load_wyckoff_symbols,
    wyckoff_symbol_to_index,
    get_wyckoff_index,
    get_wyckoff_symbols_for_sg,
)

# Helper function to get the path to the actual JSON file
# This assumes pytest is run from the project root or that the path is discoverable
def get_wyckoff_json_path():
    # Assuming the test file is in mattermake/tests/data/components
    # and the JSON is in mattermake/mattermake/data/components
    test_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(test_dir))) # Up three levels
    return os.path.join(project_root, 'mattermake', 'data', 'components', 'wyckoff_symbols.json')

# Fixture to load symbols once for all tests in this module
@pytest.fixture(scope="module")
def loaded_wyckoff_data() -> Tuple[Dict[int, List[str]], Dict[Tuple[int, str], int]]:
    # Check if the actual file exists before trying to load
    json_path = get_wyckoff_json_path()
    if not os.path.exists(json_path):
         pytest.skip(f"Wyckoff symbols JSON file not found at expected location: {json_path}")

    sg_to_symbols = load_wyckoff_symbols()
    mapping = wyckoff_symbol_to_index(sg_to_symbols)
    return sg_to_symbols, mapping

def test_load_wyckoff_symbols(loaded_wyckoff_data):
    sg_to_symbols, _ = loaded_wyckoff_data
    assert isinstance(sg_to_symbols, dict)
    assert 225 in sg_to_symbols  # Example spacegroup
    assert isinstance(sg_to_symbols[225], list)
    assert "4a" in sg_to_symbols[225] # Example symbol for sg 225

def test_wyckoff_symbol_to_index(loaded_wyckoff_data):
    _, mapping = loaded_wyckoff_data
    assert isinstance(mapping, dict)
    assert (225, "4a") in mapping
    assert isinstance(mapping[(225, "4a")], int)
    assert mapping[(225, "4a")] == 0 # Assuming '4a' is the first symbol listed for sg 225 in the JSON

def test_get_wyckoff_index(loaded_wyckoff_data):
    sg_to_symbols, mapping = loaded_wyckoff_data
    # Test a known case (using spacegroup 225, symbol '4b')
    # Find the expected index directly from the loaded symbols
    expected_index_4b = -1
    if 225 in sg_to_symbols and "4b" in sg_to_symbols[225]:
         expected_index_4b = sg_to_symbols[225].index("4b")

    assert get_wyckoff_index(225, "4b", mapping) == expected_index_4b
    assert get_wyckoff_index(225, "non_existent_symbol", mapping) == -1
    assert get_wyckoff_index(999, "4b", mapping) == -1 # Non-existent spacegroup

def test_get_wyckoff_symbols_for_sg(loaded_wyckoff_data):
    sg_to_symbols, _ = loaded_wyckoff_data
    symbols_225 = get_wyckoff_symbols_for_sg(225, sg_to_symbols)
    assert isinstance(symbols_225, list)
    assert "4a" in symbols_225
    assert "4b" in symbols_225

    # Test a potentially different spacegroup if available in JSON, e.g., 194
    if 194 in sg_to_symbols:
        symbols_194 = get_wyckoff_symbols_for_sg(194, sg_to_symbols)
        assert isinstance(symbols_194, list)
        assert "2a" in symbols_194 # Example symbol

    assert get_wyckoff_symbols_for_sg(999, sg_to_symbols) == [] # Non-existent spacegroup
