from dataclasses import dataclass


@dataclass
class HierarchicalCrystalTransformerConfig:
    """Configuration for the Hierarchical Crystal Transformer model"""

    vocab_size: int = 2000  # set based on tokenizer
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 1024
    type_vocab_size: int = 7  # Number of different segment types
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12

    space_group_embedding_dim: int = 64
    element_embedding_dim: int = 64
    wyckoff_embedding_dim: int = 32
    lattice_embedding_dim: int = 32
    coordinate_embedding_dim: int = 32

    composition_layers: int = 3
    space_group_layers: int = 2
    lattice_layers: int = 3
    atom_layers: int = 6
    integration_layers: int = 2

    use_cross_attention: bool = True
    cross_attention_heads: int = 8

    use_curriculum: bool = False
    composition_curriculum_epochs: int = 5
    space_group_curriculum_epochs: int = 5
    lattice_curriculum_epochs: int = 5

    composition_loss_weight: float = 1.2
    space_group_loss_weight: float = 1.0
    lattice_loss_weight: float = 1.0
    atom_loss_weight: float = 0.8

    # Mixture Density Network parameters
    lattice_mixture_components: int = (
        5  # Number of components for lattice parameter MoG
    )
    coord_mixture_components: int = 5  # Number of components for coordinate MoVM

    # Whether to apply Wyckoff position constraints
    apply_wyckoff_constraints: bool = False

    # Whether to use combined Wyckoff-multiplicity tokens
    use_combined_wyckoff_tokens: bool = True

    # Number of combined Wyckoff-multiplicity tokens
    # Default calculation: ~230 space groups * 26 letters = 5980 possible combinations
    num_wyckoff_mult_tokens: int = 6000

    SEGMENT_SPECIAL: int = 0
    SEGMENT_COMPOSITION: int = 1
    SEGMENT_SPACE_GROUP: int = 2
    SEGMENT_LATTICE: int = 3
    SEGMENT_ELEMENT: int = 4
    SEGMENT_WYCKOFF: int = 5
    SEGMENT_COORDINATE: int = 6
