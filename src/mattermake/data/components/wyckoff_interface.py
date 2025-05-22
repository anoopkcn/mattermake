from typing import Dict, List, Tuple, Optional
import torch
from wyckoff import WyckoffDatabase

from mattermake.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

class WyckoffInterface:
    """Interface for working with the wyckoff package."""
    
    _instance = None
    _mapping = None
    _inverse_mapping = None
    _vocab_size = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the Wyckoff database and mappings."""
        log.info("Initializing Wyckoff interface...")
        self.db = WyckoffDatabase()
        self._create_global_mapping()
        log.info(f"Created Wyckoff vocabulary with {self._vocab_size} entries")
    
    def _create_global_mapping(self):
        """Create global mapping between (sg_num, wyckoff_letter) and indices."""
        mapping = {(0, 'pad'): 0}  # Reserve 0 for padding
        inverse_mapping = {0: (0, 'pad')}
        current_idx = 1
        
        for sg_key, sg_data in self.db.data.items():
            # Extract numeric part for space groups with variants
            sg_num = int(sg_key.split('-')[0])
            
            for wyckoff_pos in sg_data.wyckoff_positions:
                key = (sg_num, wyckoff_pos.letter)
                if key not in mapping:
                    mapping[key] = current_idx
                    inverse_mapping[current_idx] = key
                    current_idx += 1
        
        self._mapping = mapping
        self._inverse_mapping = inverse_mapping
        self._vocab_size = current_idx
    
    def get_vocab_size(self) -> int:
        """Get the total vocabulary size."""
        if self._vocab_size is None:
            return 1
        return self._vocab_size
    
    def wyckoff_to_index(self, sg_num: int, letter: str) -> int:
        """Convert (space_group, wyckoff_letter) to global index."""
        if self._mapping is None:
            return 0
        key = (sg_num, letter)
        return self._mapping.get(key, 0)  # Return padding index if not found
    
    def index_to_wyckoff(self, index: int) -> Tuple[int, str]:
        """Convert global index to (space_group, wyckoff_letter)."""
        if self._inverse_mapping is None:
            return (0, 'pad')
        return self._inverse_mapping.get(index, (0, 'pad'))
    
    def get_valid_wyckoff_letters(self, sg_num: int) -> List[str]:
        """Get valid Wyckoff letters for a given space group."""
        # Find the space group in database (handle variants)
        result = self.db.find_space_group_variant(sg_num)
        
        if result is None or len(result) != 2:
            log.warning(f"Space group {sg_num} not found in database")
            return []
        
        sg_key, sg_data = result
        
        if sg_data is None:
            log.warning(f"Space group {sg_num} has no valid Wyckoff positions")
            return []
        
        # Handle both dict and object formats
        if hasattr(sg_data, 'wyckoff_positions'):
            return [wp.letter for wp in sg_data.wyckoff_positions]  # type: ignore
        elif isinstance(sg_data, dict) and 'wyckoff_positions' in sg_data:
            wyckoff_positions = sg_data['wyckoff_positions']
            if hasattr(wyckoff_positions, '__iter__'):
                return [getattr(wp, 'letter', 'a') for wp in wyckoff_positions if hasattr(wp, 'letter')]
            else:
                log.warning(f"Space group {sg_num} wyckoff_positions is not iterable")
                return []
        else:
            log.warning(f"Space group {sg_num} data format not recognized")
            return []
    
    def create_sg_mask(self, sg_numbers: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Create mask for valid Wyckoff positions per space group."""
        batch_size = sg_numbers.size(0)
        vocab_size = self._vocab_size or 1
        mask = torch.zeros(batch_size, vocab_size, dtype=torch.bool, device=device)
        
        # Always allow padding token
        mask[:, 0] = True
        
        for i, sg_num in enumerate(sg_numbers):
            sg_num_int = int(sg_num.item())
            valid_letters = self.get_valid_wyckoff_letters(sg_num_int)
            for letter in valid_letters:
                idx = self.wyckoff_to_index(sg_num_int, letter)
                mask[i, idx] = True
        
        return mask

# Global instance
wyckoff_interface = WyckoffInterface()

# Convenience functions
def get_wyckoff_vocab_size() -> int:
    return wyckoff_interface.get_vocab_size()

def wyckoff_tuple_to_index(sg_num: int, letter: str) -> int:
    return wyckoff_interface.wyckoff_to_index(sg_num, letter)

def index_to_wyckoff_tuple(index: int) -> Tuple[int, str]:
    return wyckoff_interface.index_to_wyckoff(index)

def create_wyckoff_mask(sg_numbers: torch.Tensor, device: torch.device) -> torch.Tensor:
    return wyckoff_interface.create_sg_mask(sg_numbers, device)

def get_valid_wyckoff_letters_for_sg(sg_num: int) -> List[str]:
    return wyckoff_interface.get_valid_wyckoff_letters(sg_num)