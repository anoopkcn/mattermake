import sys
import torch
import torch.nn as nn
from typing import Optional, Dict, Union, Tuple

from .base_transformer import TransformerLayer
from .attention import CrossAttentionLayer


class HierarchicalEncoder(nn.Module):
    """Base class for hierarchical encoders"""

    def __init__(self, config, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerLayer(config) for _ in range(num_layers)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_causal_mask: bool = True,
        kv_caches: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        use_kv_cache: bool = False,
        cache_prefix: str = "",
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]]:
        for i, layer in enumerate(self.layers):
            layer_cache = (
                None if kv_caches is None else kv_caches.get(f"{cache_prefix}_{i}")
            )

            if use_kv_cache and kv_caches is not None:
                hidden_states, new_cache = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    causal_mask=use_causal_mask,
                    kv_cache=layer_cache,
                    use_kv_cache=True,
                )
                kv_caches[f"{cache_prefix}_{i}"] = new_cache
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    causal_mask=use_causal_mask,
                )

        return hidden_states


class CompositionEncoder(HierarchicalEncoder):
    """Encoder for composition tokens"""

    def __init__(self, config):
        super().__init__(config, config.composition_layers)


class SpaceGroupEncoder(HierarchicalEncoder):
    """Encoder for space group tokens"""

    def __init__(self, config):
        super().__init__(config, config.space_group_layers)
        if config.use_cross_attention:
            self.cross_attention = CrossAttentionLayer(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        comp_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_causal_mask: bool = True,
        kv_caches: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        use_kv_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]]:
        # Apply cross-attention from composition if enabled
        if hasattr(self, "cross_attention"):
            # Check for NaN/Inf in hidden states before cross attention
            if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
                print(
                    "Warning: NaN/Inf detected in hidden states before space group cross-attention"
                )
                hidden_states = torch.nan_to_num(
                    hidden_states, nan=0.0, posinf=0.0, neginf=-1e5
                )

            cross_attn_cache = (
                None if kv_caches is None else kv_caches.get("sg_from_comp")
            )

            if use_kv_cache and kv_caches is not None:
                try:
                    hidden_states, new_cache = self.cross_attention(
                        hidden_states=hidden_states,
                        context_states=hidden_states,
                        attention_mask=comp_mask.unsqueeze(1).unsqueeze(2),
                        kv_cache=cross_attn_cache,
                        use_kv_cache=True,
                    )
                    # Check result for NaN/Inf
                    if (
                        torch.isnan(hidden_states).any()
                        or torch.isinf(hidden_states).any()
                    ):
                        
                        print("Warning: NaN/Inf detected after space group cross-attention", file=sys.stderr)
                        hidden_states = torch.nan_to_num(
                            hidden_states, nan=0.0, posinf=0.0, neginf=-1e5
                        )
                    kv_caches["sg_from_comp"] = new_cache
                except Exception as e:
                    print(f"Warning: Error in space group cross-attention: {e}", file=sys.stderr)
            else:
                try:
                    hidden_states_before = hidden_states.clone()
                    hidden_states = self.cross_attention(
                        hidden_states=hidden_states,
                        context_states=hidden_states,
                        attention_mask=comp_mask.unsqueeze(1).unsqueeze(2),
                    )
                    # Check result for NaN/Inf
                    if (
                        torch.isnan(hidden_states).any()
                        or torch.isinf(hidden_states).any()
                    ):
                        print("Warning: NaN/Inf detected after space group cross-attention", file=sys.stderr)
                        # Revert to input if cross attention produced NaN/Inf
                        hidden_states = hidden_states_before
                except Exception as e:
                    print(f"Warning: Error in space group cross-attention: {e}", file=sys.stderr)

        # Apply transformer layers
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            use_causal_mask=use_causal_mask,
            kv_caches=kv_caches,
            use_kv_cache=use_kv_cache,
            cache_prefix="space_group",
        )


class LatticeEncoder(HierarchicalEncoder):
    """Encoder for lattice parameter tokens"""

    def __init__(self, config):
        super().__init__(config, config.lattice_layers)
        if config.use_cross_attention:
            self.cross_attention = CrossAttentionLayer(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context_mask: torch.Tensor,  # comp_mask | sg_mask
        attention_mask: Optional[torch.Tensor] = None,
        use_causal_mask: bool = True,
        kv_caches: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        use_kv_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]]:
        # Apply cross-attention from composition and space group if enabled
        if hasattr(self, "cross_attention"):
            cross_attn_cache = (
                None if kv_caches is None else kv_caches.get("lattice_from_sg")
            )

            if use_kv_cache and kv_caches is not None:
                try:
                    hidden_states, new_cache = self.cross_attention(
                        hidden_states=hidden_states,
                        context_states=hidden_states,
                        attention_mask=context_mask,
                        kv_cache=cross_attn_cache,
                        use_kv_cache=True,
                    )
                    kv_caches["lattice_from_sg"] = new_cache
                except Exception as e:
                    print(f"Warning: Error in lattice cross-attention: {e}", file=sys.stderr)
            else:
                try:
                    hidden_states = self.cross_attention(
                        hidden_states=hidden_states,
                        context_states=hidden_states,
                        attention_mask=context_mask,
                    )
                except Exception as e:
                    print(f"Warning: Error in lattice cross-attention: {e}", file=sys.stderr)

        # Apply transformer layers
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            use_causal_mask=use_causal_mask,
            kv_caches=kv_caches,
            use_kv_cache=use_kv_cache,
            cache_prefix="lattice",
        )


class AtomEncoder(HierarchicalEncoder):
    """Encoder for atom tokens (element, Wyckoff, coordinate)"""

    def __init__(self, config):
        super().__init__(config, config.atom_layers)
        if config.use_cross_attention:
            self.cross_attention = CrossAttentionLayer(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context_mask: torch.Tensor,  # comp_mask | sg_mask | lattice_mask
        atom_mask: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_causal_mask: bool = True,
        training: bool = False,
        kv_caches: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        use_kv_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]]:
        # Apply cross-attention from composition, space group, and lattice if enabled
        if hasattr(self, "cross_attention"):
            cross_attn_cache = (
                None if kv_caches is None else kv_caches.get("atom_from_lattice")
            )

            if use_kv_cache and kv_caches is not None:
                try:
                    hidden_states, new_cache = self.cross_attention(
                        hidden_states=hidden_states,
                        context_states=hidden_states,
                        attention_mask=context_mask,
                        kv_cache=cross_attn_cache,
                        use_kv_cache=True,
                    )
                    kv_caches["atom_from_lattice"] = new_cache
                except Exception as e:
                    print(f"Warning: Error in atom cross-attention: {e}", file=sys.stderr)
            else:
                try:
                    hidden_states = self.cross_attention(
                        hidden_states=hidden_states,
                        context_states=hidden_states,
                        attention_mask=context_mask,
                    )
                except Exception as e:
                    print(f"Warning: Error in atom cross-attention: {e}", file=sys.stderr)

        # Apply atom emphasis during training
        atom_attention_mask = attention_mask
        if training and atom_mask is not None:
            atom_emphasis = 1.0 + 0.2 * atom_mask.float()
            atom_attention_mask = attention_mask * atom_emphasis.unsqueeze(1)

        # Apply transformer layers
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=atom_attention_mask if training else attention_mask,
            use_causal_mask=use_causal_mask,
            kv_caches=kv_caches,
            use_kv_cache=use_kv_cache,
            cache_prefix="atom",
        )


class IntegrationEncoder(HierarchicalEncoder):
    """Final integration layers for all hierarchical levels"""

    def __init__(self, config):
        super().__init__(config, config.integration_layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_causal_mask: bool = True,
        kv_caches: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        use_kv_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]]:
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            use_causal_mask=use_causal_mask,
            kv_caches=kv_caches,
            use_kv_cache=use_kv_cache,
            cache_prefix="integration",
        )
