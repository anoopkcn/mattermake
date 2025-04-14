import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

from .hct.base_transformer import BaseTransformer
from .hct.hierarchical import (
    CompositionEncoder,
    SpaceGroupEncoder,
    LatticeEncoder,
    AtomEncoder,
    IntegrationEncoder,
)
from .hct.generation import GenerationMixin
from .hct.loss import LossCalculationMixin
from .hct.utils import (
    bound_lattice_lengths,
    bound_lattice_angles,
    bound_fractional_coords,
)
from .hct.mixture_density import MixtureOfGaussiansHead, MixtureOfWrappedNormalsHead
from mattermake.utils.hct_wyckoff_mapping import SpaceGroupWyckoffMapping


class HierarchicalCrystalTransformer(
    BaseTransformer, GenerationMixin, LossCalculationMixin
):
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
        self._sg_wyckoff_mapping = None
        self._selected_space_groups = {}

        # Initialize Wyckoff mapping if using constraints
        if (
            hasattr(config, "apply_wyckoff_constraints")
            and config.apply_wyckoff_constraints
        ):
            self.sg_wyckoff_mapping = SpaceGroupWyckoffMapping()

        # Hierarchical encoders
        self.composition_encoder = CompositionEncoder(config)
        self.space_group_encoder = SpaceGroupEncoder(config)
        self.lattice_encoder = LatticeEncoder(config)
        self.atom_encoder = AtomEncoder(config)
        self.integration_encoder = IntegrationEncoder(config)

        # Prediction heads
        self.space_group_head = nn.Linear(config.hidden_size, 230)  # 230 space groups
        nn.init.xavier_normal_(
            self.space_group_head.weight, gain=0.1
        )  # Controlled initialization
        nn.init.constant_(self.space_group_head.bias, 0)  # Initialize bias to zero

        self.wyckoff_head = nn.Linear(
            config.hidden_size, 26
        )  # 26 Wyckoff letters (a-z)

        if (
            hasattr(config, "use_combined_wyckoff_tokens")
            and config.use_combined_wyckoff_tokens
        ):
            self.wyckoff_mult_head = nn.Linear(
                config.hidden_size, config.num_wyckoff_mult_tokens
            )  # Combined Wyckoff letter + multiplicity tokens
        else:
            self.wyckoff_mult_head = None

        self.element_head = nn.Linear(config.hidden_size, 95)  # ~95 elements

        self.lattice_mog_head = MixtureOfGaussiansHead(
            hidden_size=config.hidden_size,
            output_dim=6,  # 3 lengths + 3 angles
            n_mixtures=config.lattice_mixture_components,
        )

        self.fractional_coord_movm_head = MixtureOfWrappedNormalsHead(
            hidden_size=config.hidden_size,
            output_dim=3,  # x, y, z coordinates
            n_mixtures=config.coord_mixture_components,
        )

        # Active modules for curriculum learning
        self.active_modules = ["composition", "space_group", "lattice", "atoms"]

        # Re-apply weight initialization to ensure consistency
        self.apply(self._init_weights)

    @property
    def sg_wyckoff_mapping(self):
        """Lazy loading for SpaceGroupWyckoffMapping"""
        if (
            not hasattr(self.config, "apply_wyckoff_constraints")
            or not self.config.apply_wyckoff_constraints
        ):
            return None

        if self._sg_wyckoff_mapping is None:
            self._sg_wyckoff_mapping = SpaceGroupWyckoffMapping()
        return self._sg_wyckoff_mapping

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
        batch_size, seq_length = input_ids.size()

        embeddings = self.get_input_embeddings(input_ids, segment_ids)

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), device=input_ids.device
            )

        safe_segment_ids = torch.clamp(segment_ids, max=self.config.type_vocab_size - 1)
        comp_mask = safe_segment_ids == self.config.SEGMENT_COMPOSITION
        sg_mask = safe_segment_ids == self.config.SEGMENT_SPACE_GROUP
        lattice_mask = safe_segment_ids == self.config.SEGMENT_LATTICE
        atom_mask = (
            (safe_segment_ids == self.config.SEGMENT_ELEMENT)
            | (safe_segment_ids == self.config.SEGMENT_WYCKOFF)
            | (safe_segment_ids == self.config.SEGMENT_COORDINATE)
        )

        hidden_states = embeddings

        if "composition" in self.active_modules:
            hidden_states = self.composition_encoder(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                use_causal_mask=use_causal_mask,
                kv_caches=kv_caches,
                use_kv_cache=use_kv_cache,
                cache_prefix="composition",
            )

        if "space_group" in self.active_modules:
            hidden_states = self.space_group_encoder(
                hidden_states=hidden_states,
                comp_mask=comp_mask,
                attention_mask=attention_mask,
                use_causal_mask=use_causal_mask,
                kv_caches=kv_caches,
                use_kv_cache=use_kv_cache,
            )

        if "lattice" in self.active_modules:
            context_mask = (comp_mask | sg_mask).unsqueeze(1).unsqueeze(2)
            hidden_states = self.lattice_encoder(
                hidden_states=hidden_states,
                context_mask=context_mask,
                attention_mask=attention_mask,
                use_causal_mask=use_causal_mask,
                kv_caches=kv_caches,
                use_kv_cache=use_kv_cache,
            )

        if "atoms" in self.active_modules:
            context_mask = (
                (comp_mask | sg_mask | lattice_mask).unsqueeze(1).unsqueeze(2)
            )
            hidden_states = self.atom_encoder(
                hidden_states=hidden_states,
                context_mask=context_mask,
                atom_mask=atom_mask,
                attention_mask=attention_mask,
                use_causal_mask=use_causal_mask,
                training=self.training,
                kv_caches=kv_caches,
                use_kv_cache=use_kv_cache,
            )

        hidden_states = self.integration_encoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            use_causal_mask=use_causal_mask,
            kv_caches=kv_caches,
            use_kv_cache=use_kv_cache,
        )

        logits = self.lm_head(hidden_states)

        outputs = {"logits": logits, "hidden_states": hidden_states}

        if labels is not None:
            outputs = self.calculate_losses(
                outputs, labels, safe_segment_ids, hidden_states, batch_size
            )

        self._update_selected_space_groups(outputs, sg_mask, hidden_states)
        self._add_hierarchical_predictions(outputs, safe_segment_ids, hidden_states)

        return outputs

    def _update_selected_space_groups(self, outputs, sg_mask, hidden_states):
        """Update selected space groups for use in Wyckoff constraints"""
        if not self.training and "space_group_logits" in outputs:
            # For inference, get the most likely space group for each batch item
            batch_indices = torch.where(sg_mask)[0]
            sg_logits = outputs["space_group_logits"]

            for i, batch_idx in enumerate(batch_indices):
                if batch_idx not in self._selected_space_groups and i < sg_logits.size(
                    0
                ):
                    # +1 because space groups are 1-230, and our output is 0-229
                    space_group = sg_logits[i].argmax().item() + 1
                    if 1 <= space_group <= 230:
                        self._selected_space_groups[batch_idx.item()] = space_group

    def _add_hierarchical_predictions(self, outputs, segment_ids, hidden_states):
        """Add predictions for each hierarchical level"""
        if "space_group" in self.active_modules:
            sg_mask = segment_ids == self.config.SEGMENT_SPACE_GROUP
            if torch.any(sg_mask):
                sg_hidden = hidden_states[sg_mask]
                if sg_hidden.size(0) > 0:
                    sg_logits = self.space_group_head(sg_hidden)
                    outputs["space_group_logits"] = sg_logits

        if "lattice" in self.active_modules:
            lattice_mask = segment_ids == self.config.SEGMENT_LATTICE
            if torch.any(lattice_mask):
                lattice_hidden = hidden_states[lattice_mask]
                if lattice_hidden.size(0) > 0:
                    if (
                        torch.isnan(lattice_hidden).any()
                        or torch.isinf(lattice_hidden).any()
                    ):
                        lattice_hidden = torch.nan_to_num(
                            lattice_hidden, nan=0.0, posinf=0.0, neginf=-1e5
                        )

                    try:
                        lattice_mog_params = self.lattice_mog_head(lattice_hidden)
                        outputs["lattice_mog_params"] = lattice_mog_params

                        clean_logits = torch.nan_to_num(
                            lattice_mog_params["weights_logits"],
                            nan=0.0,
                            posinf=0.0,
                            neginf=-1e5,
                        )
                        clean_logits = (
                            clean_logits + torch.finfo(clean_logits.dtype).eps
                        )
                        weights_probs = F.softmax(clean_logits, dim=-1)

                        clean_means = torch.nan_to_num(
                            lattice_mog_params["means"], nan=0.0
                        )
                        weighted_means = torch.sum(
                            weights_probs.unsqueeze(-2) * clean_means, dim=-1
                        )
                        weighted_means = torch.nan_to_num(weighted_means, nan=0.0)

                        lattice_lengths = bound_lattice_lengths(weighted_means[:, :3])
                        lattice_angles = bound_lattice_angles(weighted_means[:, 3:])

                        outputs["lattice_lengths"] = lattice_lengths
                        outputs["lattice_angles"] = lattice_angles
                    except Exception as e:
                        print(f"Warning: Error in lattice parameter calculation: {e}", file=sys.stderr)

        if "atoms" in self.active_modules:
            element_mask = segment_ids == self.config.SEGMENT_ELEMENT
            if torch.any(element_mask):
                element_hidden = hidden_states[element_mask]
                if element_hidden.size(0) > 0:
                    element_logits = self.element_head(element_hidden)
                    outputs["element_logits"] = element_logits

            wyckoff_mask = segment_ids == self.config.SEGMENT_WYCKOFF
            if torch.any(wyckoff_mask):
                wyckoff_hidden = hidden_states[wyckoff_mask]
                if wyckoff_hidden.size(0) > 0:
                    batch_indices = torch.where(wyckoff_mask)[0]

                    all_wyckoff_logits = []

                    for i, batch_idx in enumerate(batch_indices):
                        batch_idx = batch_idx.item()

                        use_combined = (
                            hasattr(self.config, "use_combined_wyckoff_tokens")
                            and self.config.use_combined_wyckoff_tokens
                            and self.wyckoff_mult_head is not None
                        )

                        if use_combined:
                            wyckoff_logits_i = self.wyckoff_mult_head(
                                wyckoff_hidden[i : i + 1]
                            )
                        else:
                            wyckoff_logits_i = self.wyckoff_head(
                                wyckoff_hidden[i : i + 1]
                            )

                        if (
                            hasattr(self.config, "apply_wyckoff_constraints")
                            and self.config.apply_wyckoff_constraints
                            and batch_idx in self._selected_space_groups
                            and self._sg_wyckoff_mapping is not None
                        ):
                            space_group = self._selected_space_groups[batch_idx]

                            if use_combined:
                                allowed_wyckoff = self._sg_wyckoff_mapping.get_allowed_wyckoff_positions(
                                    space_group
                                )

                                valid_mask = torch.zeros(
                                    self.config.num_wyckoff_mult_tokens,
                                    device=wyckoff_hidden.device,
                                    dtype=torch.bool,
                                )

                                offset = 3000 + (
                                    space_group * 100
                                )  # Combined token range starts at 3000
                                for letter_idx, letter in enumerate(allowed_wyckoff):
                                    idx = ord(letter) - ord("a")
                                    if 0 <= idx < 26:
                                        token_id = offset + idx
                                        if token_id < valid_mask.size(0):
                                            valid_mask[token_id] = True
                            else:
                                valid_mask = torch.tensor(
                                    self._sg_wyckoff_mapping.create_wyckoff_mask(
                                        space_group
                                    ),
                                    device=wyckoff_hidden.device,
                                    dtype=torch.bool,
                                )

                            if (
                                valid_mask is not None
                                and valid_mask.shape[0] <= wyckoff_logits_i.shape[1]
                            ):
                                wyckoff_logits_i[:, ~valid_mask] = float("-inf")

                        all_wyckoff_logits.append(wyckoff_logits_i)

                    if all_wyckoff_logits:
                        wyckoff_logits = torch.cat(all_wyckoff_logits, dim=0)
                        outputs["wyckoff_logits"] = wyckoff_logits

            coordinate_mask = segment_ids == self.config.SEGMENT_COORDINATE
            if torch.any(coordinate_mask):
                coordinate_hidden = hidden_states[coordinate_mask]
                if coordinate_hidden.size(0) > 0:
                    outputs["coordinate_hidden"] = coordinate_hidden

                    try:
                        if (
                            torch.isnan(coordinate_hidden).any()
                            or torch.isinf(coordinate_hidden).any()
                        ):
                            coordinate_hidden = torch.nan_to_num(
                                coordinate_hidden, nan=0.0, posinf=0.0, neginf=-1e5
                            )

                        num_atoms = coordinate_hidden.size(0) // 3

                        if coordinate_hidden.size(0) % 3 != 0:
                            num_atoms = coordinate_hidden.size(0) // 3

                        if num_atoms > 0 and coordinate_hidden.size(0) >= 3:
                            slice_end = num_atoms * 3
                            if slice_end <= coordinate_hidden.size(0):
                                coord_input = coordinate_hidden[:slice_end:3]

                                coord_movm_params = self.fractional_coord_movm_head(
                                    coord_input
                                )
                                outputs["coord_movm_params"] = coord_movm_params

                                clean_logits = torch.nan_to_num(
                                    coord_movm_params["weights_logits"],
                                    nan=0.0,
                                    posinf=0.0,
                                    neginf=-1e5,
                                )

                                clean_logits = (
                                    clean_logits + torch.finfo(clean_logits.dtype).eps
                                )

                                weights_probs = F.softmax(clean_logits, dim=-1)

                                clean_means = torch.nan_to_num(
                                    coord_movm_params["means"], nan=0.5
                                )  # Default to middle of [0,1) range

                                weighted_means = torch.sum(
                                    weights_probs.unsqueeze(-2) * clean_means,
                                    dim=-1,
                                )

                                weighted_means = torch.nan_to_num(
                                    weighted_means, nan=0.5
                                )

                                # Bound coordinates to [0, 1)
                                from .hct.utils import bound_fractional_coords

                                fractional_coords = bound_fractional_coords(
                                    weighted_means
                                )

                                outputs["fractional_coords"] = fractional_coords
                    except Exception as e:
                        print(f"Warning: Error in coordinate prediction: {e}")

        # For generation, ensure lattice and coordinate predictions are included
        # even if the right segments weren't encountered during this forward pass
        if not self.training:
            self._ensure_continuous_predictions(outputs, hidden_states)

        return outputs

    def _ensure_continuous_predictions(self, outputs, hidden_states):
        """Ensure lattice and coordinate predictions are available for generation"""
        if "lattice" in self.active_modules and "lattice_lengths" not in outputs:
            try:
                last_hidden = hidden_states[:, -1].unsqueeze(0)
                lattice_mog_params = self.lattice_mog_head(last_hidden)
                outputs["lattice_mog_params"] = lattice_mog_params

                clean_logits = torch.nan_to_num(
                    lattice_mog_params["weights_logits"],
                    nan=0.0,
                    posinf=0.0,
                    neginf=-1e5,
                )
                clean_logits = clean_logits + torch.finfo(clean_logits.dtype).eps
                weights_probs = F.softmax(clean_logits, dim=-1)

                clean_means = torch.nan_to_num(lattice_mog_params["means"], nan=0.0)
                weighted_means = torch.sum(
                    weights_probs.unsqueeze(-2) * clean_means, dim=-1
                )
                weighted_means = torch.nan_to_num(weighted_means, nan=0.0)

                # Calculate bounded lattice parameters
                lattice_lengths = bound_lattice_lengths(weighted_means[:, :3])
                lattice_angles = bound_lattice_angles(weighted_means[:, 3:])

                outputs["lattice_lengths"] = lattice_lengths
                outputs["lattice_angles"] = lattice_angles
            except Exception as e:
                print(f"Warning: Error ensuring lattice predictions: {e}")
                device = hidden_states.device
                batch_size = hidden_states.size(0)
                outputs["lattice_lengths"] = (
                    torch.ones((batch_size, 3), device=device) * 5.0
                )
                outputs["lattice_angles"] = (
                    torch.ones((batch_size, 3), device=device) * 90.0
                )

        if "atoms" in self.active_modules and "fractional_coords" not in outputs:
            try:
                last_hidden = hidden_states[:, -1].unsqueeze(0)

                coord_movm_params = self.fractional_coord_movm_head(last_hidden)
                outputs["coord_movm_params"] = coord_movm_params

                clean_logits = torch.nan_to_num(
                    coord_movm_params["weights_logits"],
                    nan=0.0,
                    posinf=0.0,
                    neginf=-1e5,
                )
                clean_logits = clean_logits + torch.finfo(clean_logits.dtype).eps
                weights_probs = F.softmax(clean_logits, dim=-1)

                clean_means = torch.nan_to_num(coord_movm_params["means"], nan=0.5)
                weighted_means = torch.sum(
                    weights_probs.unsqueeze(-2) * clean_means, dim=-1
                )
                weighted_means = torch.nan_to_num(weighted_means, nan=0.5)

                fractional_coords = bound_fractional_coords(weighted_means)

                outputs["fractional_coords"] = fractional_coords
            except Exception as e:
                print(f"Warning: Error ensuring coordinate predictions: {e}")
                device = hidden_states.device
                batch_size = hidden_states.size(0)
                outputs["fractional_coords"] = (
                    torch.ones((batch_size, 3), device=device) * 0.5
                )

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
