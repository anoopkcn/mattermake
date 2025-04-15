import torch
import torch.nn as nn
from typing import Optional, Dict

from mattermake.models.components.hct_utils.base_transformer import BaseTransformer
from mattermake.models.components.hct_utils.transformer_encoder import TransformerEncoder
from mattermake.models.components.hct_utils.cross_attention import CrossAttention
from mattermake.models.components.hct_utils.generation import GenerationMixin
from mattermake.models.components.hct_utils.loss import LossCalculationMixin
from mattermake.models.components.hct_utils.mixture_density import (
    SimplifiedMixtureOfGaussians,
    SimplifiedMixtureOfWrappedNormals,
    bound_lattice_lengths,
    bound_lattice_angles,
    bound_fractional_coords
)
from mattermake.utils.hct_wyckoff_mapping import SpaceGroupWyckoffMapping


class HierarchicalCrystalTransformer(BaseTransformer, GenerationMixin, LossCalculationMixin):
    """Hierarchical transformer for coarse-to-fine crystal structure generation

    This uses a modular architecture with:
    1. Composition encoder
    2. Space group encoder
    3. Lattice encoder
    4. Atom encoder (element, wyckoff, coordinate)
    5. Integration layers

    Each hierarchical level uses information from previous levels through cross-attention.
    """

    def __init__(self, config):
        BaseTransformer.__init__(self, config)
        self.config = config
        
        # Core configuration and segment constants
        self.segment_composition = config.SEGMENT_COMPOSITION
        self.segment_space_group = config.SEGMENT_SPACE_GROUP
        self.segment_lattice = config.SEGMENT_LATTICE
        self.segment_element = config.SEGMENT_ELEMENT
        self.segment_wyckoff = config.SEGMENT_WYCKOFF
        self.segment_coordinate = config.SEGMENT_COORDINATE
        
        # Simplify state tracking
        self._selected_space_groups = {}
        
        # Wyckoff mapping initialization (only when needed)
        self._sg_wyckoff_mapping = SpaceGroupWyckoffMapping() if hasattr(config, "apply_wyckoff_constraints") and config.apply_wyckoff_constraints else None
        
        # Simplified encoder architecture - fewer layers with clearer responsibilities
        self.encoders = nn.ModuleDict({
            'composition': TransformerEncoder(config, num_layers=config.composition_layers),
            'space_group': TransformerEncoder(config, num_layers=config.space_group_layers),
            'lattice': TransformerEncoder(config, num_layers=config.lattice_layers),
            'atom': TransformerEncoder(config, num_layers=config.atom_layers),
            'integration': TransformerEncoder(config, num_layers=config.integration_layers)
        })
        
        # Cross-attention modules for information flow between hierarchical levels
        use_cross_attention = hasattr(config, "use_cross_attention") and config.use_cross_attention
        self.cross_attentions = nn.ModuleDict({
            'sg_from_comp': CrossAttention(config) if use_cross_attention else None,
            'lattice_from_sg': CrossAttention(config) if use_cross_attention else None,
            'atom_from_lattice': CrossAttention(config) if use_cross_attention else None
        })
        
        # Simplified prediction heads with standardized outputs
        self.heads = nn.ModuleDict({
            'space_group': nn.Linear(config.hidden_size, 230),  # 230 space groups
            'element': nn.Linear(config.hidden_size, 95),     # ~95 elements
            'wyckoff': nn.Linear(config.hidden_size, 26)      # 26 Wyckoff letters
        })
        
        # Optional combined Wyckoff-multiplicity head
        if hasattr(config, "use_combined_wyckoff_tokens") and config.use_combined_wyckoff_tokens:
            self.heads['wyckoff_mult'] = nn.Linear(
                config.hidden_size, 
                config.num_wyckoff_mult_tokens if hasattr(config, "num_wyckoff_mult_tokens") else 6000
            )
        
        # New streamlined MDN implementation
        self.continuous_heads = nn.ModuleDict({
            'lattice': SimplifiedMixtureOfGaussians(
                hidden_size=config.hidden_size,
                output_dim=6,  # 3 lengths + 3 angles
                n_mixtures=config.lattice_mixture_components if hasattr(config, "lattice_mixture_components") else 5
            ),
            'coordinate': SimplifiedMixtureOfWrappedNormals(
                hidden_size=config.hidden_size,
                output_dim=3,  # x, y, z coordinates
                n_mixtures=config.coord_mixture_components if hasattr(config, "coord_mixture_components") else 5
            )
        })
        
        # Output projection
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Active modules tracker
        self.active_modules = ["composition", "space_group", "lattice", "atoms"]
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def set_active_modules(self, module_list):
        """Set which modules are active (for curriculum learning)"""
        self.active_modules = module_list
    
    def set_ground_truth_values(self, lattice_params=None, fractional_coords=None):
        """Set ground truth values for continuous prediction regression loss

        Args:
            lattice_params: Dictionary with 'lengths' and 'angles' tensors for lattice parameters
            fractional_coords: Dictionary with 'fractional_coords' tensor of shape [num_atoms, 3]
        """
        if lattice_params is not None:
            self._lattice_ground_truth = lattice_params

        if fractional_coords is not None:
            self._coordinate_ground_truth = fractional_coords
    
    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_causal_mask: bool = True,
        return_dict: bool = True,
        kv_caches: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        use_kv_cache: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with consolidated execution paths and consistent shapes"""
        batch_size, seq_length = input_ids.size()
        
        # Get embeddings
        embeddings = self.get_input_embeddings(input_ids, segment_ids)
        
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=input_ids.device)

        # Create segment masks for different token types
        safe_segment_ids = torch.clamp(segment_ids, max=self.config.type_vocab_size - 1)
        comp_mask = safe_segment_ids == self.segment_composition
        sg_mask = safe_segment_ids == self.segment_space_group
        lattice_mask = safe_segment_ids == self.segment_lattice
        element_mask = safe_segment_ids == self.segment_element
        wyckoff_mask = safe_segment_ids == self.segment_wyckoff
        
        # Process through hierarchical encoders in sequence
        hidden_states = embeddings
        outputs = {"hidden_states": {}}
        
        # 1. Composition Encoding
        if "composition" in self.active_modules:
            hidden_states = self.encoders['composition'](
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                use_causal_mask=use_causal_mask,
                kv_caches=kv_caches,
                cache_prefix="composition"
            )
            outputs["hidden_states"]["composition"] = hidden_states
        
        # 2. Space Group Encoding with cross-attention from composition
        if "space_group" in self.active_modules:
            # Apply cross-attention if available
            if self.cross_attentions['sg_from_comp'] is not None:
                sg_context_mask = comp_mask.unsqueeze(1).unsqueeze(2)
                hidden_states = self.cross_attentions['sg_from_comp'](
                    hidden_states=hidden_states,
                    context_states=hidden_states,
                    attention_mask=sg_context_mask
                )
            
            # Process through space group encoder
            hidden_states = self.encoders['space_group'](
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                use_causal_mask=use_causal_mask,
                kv_caches=kv_caches,
                cache_prefix="space_group"
            )
            outputs["hidden_states"]["space_group"] = hidden_states
            
            # Space group prediction (extraction and head application)
            if torch.any(sg_mask):
                sg_hidden = self._extract_segment_hidden(hidden_states, sg_mask)
                if sg_hidden.size(0) > 0:
                    outputs["space_group_logits"] = self.heads['space_group'](sg_hidden)
        
        # 3. Lattice Encoding with cross-attention from composition and space group
        if "lattice" in self.active_modules:
            # Apply cross-attention if available
            if self.cross_attentions['lattice_from_sg'] is not None:
                context_mask = (comp_mask | sg_mask).unsqueeze(1).unsqueeze(2)
                hidden_states = self.cross_attentions['lattice_from_sg'](
                    hidden_states=hidden_states,
                    context_states=hidden_states,
                    attention_mask=context_mask
                )
            
            # Process through lattice encoder
            hidden_states = self.encoders['lattice'](
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                use_causal_mask=use_causal_mask,
                kv_caches=kv_caches,
                cache_prefix="lattice"
            )
            outputs["hidden_states"]["lattice"] = hidden_states
        
        # 4. Atom Encoding with cross-attention from all previous levels
        if "atoms" in self.active_modules:
            # Apply cross-attention if available
            if self.cross_attentions['atom_from_lattice'] is not None:
                context_mask = (comp_mask | sg_mask | lattice_mask).unsqueeze(1).unsqueeze(2)
                hidden_states = self.cross_attentions['atom_from_lattice'](
                    hidden_states=hidden_states,
                    context_states=hidden_states,
                    attention_mask=context_mask
                )
            
            # Process through atom encoder
            hidden_states = self.encoders['atom'](
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                use_causal_mask=use_causal_mask,
                kv_caches=kv_caches,
                cache_prefix="atom"
            )
            outputs["hidden_states"]["atom"] = hidden_states
            
            # Element prediction
            if torch.any(element_mask):
                element_hidden = self._extract_segment_hidden(hidden_states, element_mask)
                if element_hidden.size(0) > 0:
                    outputs["element_logits"] = self.heads['element'](element_hidden)
            
            # Wyckoff prediction with constraints
            if torch.any(wyckoff_mask):
                wyckoff_hidden = self._extract_segment_hidden(hidden_states, wyckoff_mask)
                if wyckoff_hidden.size(0) > 0:
                    # Use combined or regular wyckoff head based on config
                    use_combined = (
                        hasattr(self.config, "use_combined_wyckoff_tokens")
                        and self.config.use_combined_wyckoff_tokens
                        and 'wyckoff_mult' in self.heads
                    )
                    
                    if use_combined:
                        wyckoff_logits = self.heads['wyckoff_mult'](wyckoff_hidden)
                    else:
                        wyckoff_logits = self.heads['wyckoff'](wyckoff_hidden)
                    
                    # Apply space group constraints if needed
                    wyckoff_logits = self._apply_wyckoff_constraints(wyckoff_logits)
                    outputs["wyckoff_logits"] = wyckoff_logits
        
        # 5. Final integration
        hidden_states = self.encoders['integration'](
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            use_causal_mask=use_causal_mask,
            kv_caches=kv_caches,
            cache_prefix="integration"
        )
        
        # Generate full logits for next token prediction
        logits = self.lm_head(hidden_states)
        outputs["logits"] = logits
        outputs["hidden_states"]["final"] = hidden_states
        
        # Add continuous predictions with standardized shapes
        self._add_continuous_predictions(outputs, hidden_states, safe_segment_ids)
        
        # Update selected space groups for use in constraints
        self._update_selected_space_groups(outputs, sg_mask, hidden_states)
        
        # Calculate losses if training
        if labels is not None:
            outputs = self.calculate_losses(outputs, labels, safe_segment_ids, hidden_states, batch_size)
        
        return outputs
    
    def _extract_segment_hidden(self, hidden_states, segment_mask):
        """Extract hidden states for a specific segment with shape validation"""
        # Extract hidden states for tokens of the specified segment
        segment_hidden = hidden_states[segment_mask]
        
        # Handle empty case gracefully
        if segment_hidden.size(0) == 0:
            return torch.zeros((0, hidden_states.size(-1)), device=hidden_states.device)
        
        return segment_hidden
    
    def _update_selected_space_groups(self, outputs, sg_mask, hidden_states):
        """Update selected space groups for use in Wyckoff constraints"""
        if not self.training and "space_group_logits" in outputs:
            # For inference, get the most likely space group for each batch item
            batch_indices = torch.where(sg_mask)[0]
            sg_logits = outputs["space_group_logits"]

            for i, batch_idx in enumerate(batch_indices):
                if batch_idx not in self._selected_space_groups and i < sg_logits.size(0):
                    # +1 because space groups are 1-230, and our output is 0-229
                    space_group = sg_logits[i].argmax().item() + 1
                    if 1 <= space_group <= 230:
                        self._selected_space_groups[batch_idx.item()] = space_group
    
    def _apply_wyckoff_constraints(self, wyckoff_logits):
        """Apply Wyckoff position constraints with explicit shape handling"""
        if not hasattr(self.config, "apply_wyckoff_constraints") or not self.config.apply_wyckoff_constraints:
            return wyckoff_logits
        
        if self._sg_wyckoff_mapping is None:
            return wyckoff_logits
        
        # Apply constraints for each batch item
        constrained_logits = wyckoff_logits.clone()
        batch_size = wyckoff_logits.size(0)
        
        for i in range(batch_size):
            # Skip if no space group selected for this batch item
            if i not in self._selected_space_groups:
                continue
                
            space_group = self._selected_space_groups[i]
            
            # Get the allowed Wyckoff positions mask
            use_combined = (
                hasattr(self.config, "use_combined_wyckoff_tokens")
                and self.config.use_combined_wyckoff_tokens
                and 'wyckoff_mult' in self.heads
            )
            
            if use_combined:
                # Create a mask for combined tokens
                allowed_wyckoff = self._sg_wyckoff_mapping.get_allowed_wyckoff_positions(space_group)
                valid_mask = torch.zeros(
                    self.heads['wyckoff_mult'].out_features,
                    device=wyckoff_logits.device,
                    dtype=torch.bool
                )
                
                # Set valid positions based on the tokens for this space group
                offset = 3000 + (space_group * 100)
                for letter_idx, letter in enumerate(allowed_wyckoff):
                    idx = ord(letter) - ord('a')
                    if 0 <= idx < 26:
                        token_id = offset + idx
                        if token_id < valid_mask.size(0):
                            valid_mask[token_id] = True
            else:
                # Regular Wyckoff position mask
                valid_mask = torch.tensor(
                    self._sg_wyckoff_mapping.create_wyckoff_mask(space_group),
                    device=wyckoff_logits.device,
                    dtype=torch.bool
                )
            
            # Ensure mask size matches logits
            if valid_mask.shape[0] <= wyckoff_logits.shape[1]:
                # Set invalid positions to -inf
                constrained_logits[i, ~valid_mask] = float("-inf")
        
        return constrained_logits
    
    def _add_continuous_predictions(self, outputs, hidden_states, segment_ids):
        """Add continuous predictions with standardized shapes and consistent error handling"""
        
        # 1. Lattice parameter prediction
        if "lattice" in self.active_modules:
            lattice_mask = segment_ids == self.segment_lattice
            if torch.any(lattice_mask):
                lattice_hidden = self._extract_segment_hidden(hidden_states, lattice_mask)
                
                # Process if we have valid hidden states
                if lattice_hidden.size(0) > 0:
                    try:
                        # Get mixture parameters with standardized shapes
                        lattice_params = self.continuous_heads['lattice'](lattice_hidden)
                        outputs["lattice_mog_params"] = lattice_params
                        
                        # Calculate weighted means with explicit shape handling
                        weighted_means = self.continuous_heads['lattice'].compute_weighted_mean(lattice_params)
                        
                        # Convert to bounded lattice parameters
                        outputs["lattice_lengths"] = bound_lattice_lengths(weighted_means[:, :3])
                        outputs["lattice_angles"] = bound_lattice_angles(weighted_means[:, 3:])
                    except Exception:
                        # Fallback with clear shapes
                        batch_size = lattice_hidden.size(0)
                        device = lattice_hidden.device
                        outputs["lattice_lengths"] = torch.ones((batch_size, 3), device=device) * 5.0
                        outputs["lattice_angles"] = torch.ones((batch_size, 3), device=device) * 90.0
        
        # 2. Coordinate prediction
        if "atoms" in self.active_modules:
            coord_mask = segment_ids == self.segment_coordinate
            if torch.any(coord_mask):
                coord_hidden = self._extract_segment_hidden(hidden_states, coord_mask)
                
                if coord_hidden.size(0) > 0:
                    try:
                        # Calculate number of atoms based on coordinate tokens
                        num_atoms = coord_hidden.size(0) // 3
                        
                        # Safety check: ensure we have complete atoms (multiples of 3 tokens)
                        if coord_hidden.size(0) % 3 != 0:
                            num_atoms = coord_hidden.size(0) // 3
                        
                        # Process coordinates if we have at least one atom
                        if num_atoms > 0 and coord_hidden.size(0) >= 3:
                            # Take every third element for coordinate prediction
                            slice_end = num_atoms * 3
                            coord_input = coord_hidden[:slice_end:3]
                            
                            # Get mixture parameters
                            coord_params = self.continuous_heads['coordinate'](coord_input)
                            outputs["coord_movm_params"] = coord_params
                            
                            # Calculate weighted means
                            weighted_means = self.continuous_heads['coordinate'].compute_weighted_mean(coord_params)
                            
                            # Convert to bounded fractional coordinates
                            outputs["fractional_coords"] = bound_fractional_coords(weighted_means)
                    except Exception:
                        # Fallback with clear shapes
                        batch_size = 1 if not isinstance(coord_hidden, torch.Tensor) else coord_hidden.size(0) // 3
                        device = hidden_states.device
                        outputs["fractional_coords"] = torch.rand((batch_size, 3), device=device)
        
        # 3. Ensure predictions exist for generation
        if not self.training:
            self._ensure_all_predictions(outputs, hidden_states)
    
    def _ensure_all_predictions(self, outputs, hidden_states):
        """Ensure all necessary predictions exist for generation with consistent shapes"""
        batch_size = hidden_states.size(0)
        device = hidden_states.device
        
        # Ensure lattice parameters exist
        if ("lattice" in self.active_modules and 
            ("lattice_lengths" not in outputs or "lattice_angles" not in outputs)):
            # Use last hidden state as input
            last_hidden = hidden_states[:, -1]
            
            try:
                # Get mixture parameters
                lattice_params = self.continuous_heads['lattice'](last_hidden)
                
                # Calculate weighted means
                weighted_means = self.continuous_heads['lattice'].compute_weighted_mean(lattice_params)
                
                # Convert to bounded lattice parameters
                outputs["lattice_lengths"] = bound_lattice_lengths(weighted_means[:, :3])
                outputs["lattice_angles"] = bound_lattice_angles(weighted_means[:, 3:])
            except Exception:
                # Clear fallback with explicit shapes
                outputs["lattice_lengths"] = torch.ones((batch_size, 3), device=device) * 5.0
                outputs["lattice_angles"] = torch.ones((batch_size, 3), device=device) * 90.0
        
        # Ensure fractional coordinates exist
        if ("atoms" in self.active_modules and "fractional_coords" not in outputs):
            try:
                # Use last hidden state for coordinate prediction
                last_hidden = hidden_states[:, -1]
                
                # Get mixture parameters
                coord_params = self.continuous_heads['coordinate'](last_hidden)
                
                # Calculate weighted means
                weighted_means = self.continuous_heads['coordinate'].compute_weighted_mean(coord_params)
                
                # Convert to bounded fractional coordinates
                outputs["fractional_coords"] = bound_fractional_coords(weighted_means)
            except Exception:
                # Use fallback random coordinates
                outputs["fractional_coords"] = torch.rand((batch_size, 3), device=device)
    
    def _predict_next_segment_id(self, sequence_ids, segment_ids, next_tokens):
        """Predict the segment ID for the next token based on context"""
        batch_size = sequence_ids.shape[0]

        current_segment = segment_ids[:, -1]
        next_segments = current_segment.clone()

        for i in range(batch_size):
            token_id = next_tokens[i].item()
            curr_seg = current_segment[i].item()

            seq = sequence_ids[i]
            segs = segment_ids[i]

            if curr_seg == self.config.SEGMENT_COMPOSITION:
                if token_id == 3:  # COMP_SEP_TOKEN
                    next_segments[i] = self.config.SEGMENT_SPECIAL

            elif curr_seg == self.config.SEGMENT_SPECIAL:
                comp_sep_positions = (seq == 3).nonzero().squeeze(-1)
                if (
                    len(comp_sep_positions) > 0
                    and segs[-1] == self.config.SEGMENT_SPECIAL
                ):
                    next_segments[i] = self.config.SEGMENT_SPACE_GROUP

            elif curr_seg == self.config.SEGMENT_SPACE_GROUP:
                next_segments[i] = self.config.SEGMENT_LATTICE

            elif curr_seg == self.config.SEGMENT_LATTICE:
                lattice_positions = (segs == self.config.SEGMENT_LATTICE).sum().item()
                if (
                    lattice_positions >= 6
                ):  # 6 lattice parameters (a, b, c, alpha, beta, gamma)
                    next_segments[i] = self.config.SEGMENT_ELEMENT

            elif curr_seg == self.config.SEGMENT_ELEMENT:
                next_segments[i] = self.config.SEGMENT_WYCKOFF

            elif curr_seg == self.config.SEGMENT_WYCKOFF:
                next_segments[i] = self.config.SEGMENT_COORDINATE

            elif curr_seg == self.config.SEGMENT_COORDINATE:
                coord_count = 0
                for j in range(len(segs) - 1, -1, -1):
                    if segs[j] == self.config.SEGMENT_COORDINATE:
                        coord_count += 1
                    else:
                        break

                if coord_count % 3 == 0:  # After 3 coordinates (x, y, z)
                    next_segments[i] = self.config.SEGMENT_ELEMENT

        return next_segments.unsqueeze(-1)