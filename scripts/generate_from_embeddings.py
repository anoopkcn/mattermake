"""
Embedding-based crystal structure generation for cross-modal synthesis.

This module enables generating crystal structures from pre-computed embeddings
created by binding models that unify different modalities (PXRD, DOS, structure)
into a shared embedding space.

Example usage:
    # From PXRD data
    python generate_from_embeddings.py \
        --checkpoint models/best_model.ckpt \
        --embedding_file pxrd_embeddings.pt \
        --embedding_type pxrd \
        --num_structures 5

    # From DOS data  
    python generate_from_embeddings.py \
        --checkpoint models/best_model.ckpt \
        --embedding_file dos_embeddings.pt \
        --embedding_type dos \
        --num_structures 3
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import rootutils

# Setup root path
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from mattermake.models.modular_hierarchical_crystal_transformer_module import (
    ModularHierarchicalCrystalTransformer,
)
from mattermake.models.components.modular_encoders import EncoderBase
from mattermake.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class EmbeddingEncoder(EncoderBase):
    """
    Encoder that processes pre-computed embeddings from binding models.
    
    This encoder accepts embeddings from various modalities (PXRD, DOS, etc.)
    and projects them to the model's expected dimensionality.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        d_model: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        has_conditioning: bool = False,
    ):
        """
        Initialize the embedding encoder.
        
        Args:
            embedding_dim: Dimensionality of input embeddings
            d_model: Output dimensionality (model's hidden dimension)
            num_layers: Number of projection layers
            dropout: Dropout rate
            has_conditioning: Whether this encoder supports conditioning
        """
        super().__init__(d_output=d_model)
        
        # Build projection network
        layers = []
        current_dim = embedding_dim
        
        for i in range(num_layers):
            target_dim = d_model if i == num_layers - 1 else (embedding_dim + d_model) // 2
            layers.extend([
                nn.Linear(current_dim, target_dim),
                nn.LayerNorm(target_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = target_dim
        
        # Remove last ReLU and dropout
        layers = layers[:-2]
        self.projector = nn.Sequential(*layers)
        
        # Setup conditioning if needed
        if has_conditioning:
            self.condition_projector = nn.Linear(d_model, d_model)
    
    def forward(
        self, 
        x: torch.Tensor, 
        condition_context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for embedding encoder.
        
        Args:
            x: Input embeddings (batch_size, embedding_dim) or (batch_size, seq_len, embedding_dim)
            condition_context: Optional conditioning context
            
        Returns:
            Projected embeddings (batch_size, 1, d_model) or (batch_size, seq_len, d_model)
        """
        # Handle both 2D and 3D inputs
        if x.dim() == 2:
            # (batch_size, embedding_dim) -> (batch_size, 1, embedding_dim)
            x = x.unsqueeze(1)
        
        # Project embeddings
        projected = self.projector(x)  # (batch_size, seq_len, d_model)
        
        # Apply conditioning if available
        if condition_context is not None and self.condition_projector is not None:
            condition_proj = self.condition_projector(condition_context)
            projected = projected + condition_proj
        
        return projected


class EmbeddingBasedGenerator:
    """
    Generator that creates crystal structures from pre-computed embeddings.
    
    This class modifies the standard MatterMake model to accept embeddings
    from binding models instead of raw composition data.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        embedding_dim: int,
        device: str = "auto",
        embedding_encoder_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the embedding-based generator.
        
        Args:
            checkpoint_path: Path to trained MatterMake model checkpoint
            embedding_dim: Dimensionality of input embeddings
            device: Device for inference
            embedding_encoder_config: Configuration for embedding encoder
        """
        self.checkpoint_path = checkpoint_path
        self.embedding_dim = embedding_dim
        self.device = self._setup_device(device)
        
        # Default embedding encoder config
        self.embedding_encoder_config = embedding_encoder_config or {
            "num_layers": 3,
            "dropout": 0.1,
            "has_conditioning": False
        }
        
        # Load and modify the model
        self.model = self._load_and_modify_model()
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def _load_and_modify_model(self) -> ModularHierarchicalCrystalTransformer:
        """Load the MatterMake model and replace composition encoder with embedding encoder."""
        log.info(f"Loading model from checkpoint: {self.checkpoint_path}")
        
        try:
            # Load the trained model
            model = ModularHierarchicalCrystalTransformer.load_from_checkpoint(
                self.checkpoint_path,
                map_location=self.device
            )
            
            # Get the model's d_model parameter from the composition encoder
            composition_encoder = model.model.encoders['composition']
            if hasattr(composition_encoder, 'd_output'):
                d_output = composition_encoder.d_output
                if isinstance(d_output, int):
                    d_model = d_output
                elif isinstance(d_output, torch.Tensor):
                    # Handle tensor case
                    d_model = int(d_output.item())
                elif hasattr(d_output, '__int__') and not isinstance(d_output, torch.nn.Module):
                    # Handle objects that can be converted to int (but not modules)
                    d_model = int(d_output)
                else:
                    # Fallback to a reasonable default
                    log.warning(f"Cannot convert d_output to int: {type(d_output)}, using default")
                    d_model = 256
            else:
                # Fallback to a reasonable default
                d_model = 256
            
            # Create new embedding encoder
            embedding_encoder = EmbeddingEncoder(
                embedding_dim=self.embedding_dim,
                d_model=d_model,
                **self.embedding_encoder_config
            )
            
            # Replace the composition encoder
            model.model.encoders['composition'] = embedding_encoder
            
            # Move to device and set to eval mode
            model.to(self.device)
            model.eval()
            
            log.info(f"Model modified successfully. Embedding dim: {self.embedding_dim} -> d_model: {d_model}")
            return model
            
        except Exception as e:
            log.error(f"Failed to load and modify model: {e}")
            raise
    
    def generate_from_embedding(
        self,
        embedding: torch.Tensor,
        num_structures: int = 5,
        max_atoms: int = 50,
        spacegroup: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        sg_sampling_mode: str = "sample",
        lattice_sampling_mode: str = "sample",
        atom_discrete_sampling_mode: str = "sample",
        coord_sampling_mode: str = "sample",
    ) -> List[Dict[str, Any]]:
        """
        Generate crystal structures from a pre-computed embedding.
        
        Args:
            embedding: Pre-computed embedding tensor (embedding_dim,) or (1, embedding_dim)
            num_structures: Number of structures to generate
            max_atoms: Maximum atoms per structure
            spacegroup: Optional fixed space group
            temperature: Sampling temperature
            *_sampling_mode: Sampling modes for different components
            
        Returns:
            List of generated crystal structures
        """
        log.info(f"Generating {num_structures} structures from embedding")
        log.info(f"Embedding shape: {embedding.shape}, max_atoms: {max_atoms}")
        
        # Ensure embedding has batch dimension
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)  # (1, embedding_dim)
        
        generated_structures = []
        
        with torch.no_grad():
            for i in range(num_structures):
                try:
                    # Move embedding to device
                    emb_tensor = embedding.to(self.device)
                    sg_tensor = spacegroup.to(self.device) if spacegroup is not None else None
                    
                    # Generate structure using embedding as "composition"
                    result = self.model.generate(
                        composition=emb_tensor,  # Pass embedding as composition
                        spacegroup=sg_tensor,
                        max_atoms=max_atoms,
                        sg_sampling_mode=sg_sampling_mode,
                        lattice_sampling_mode=lattice_sampling_mode,
                        atom_discrete_sampling_mode=atom_discrete_sampling_mode,
                        coord_sampling_mode=coord_sampling_mode,
                        temperature=temperature,
                    )
                    
                    # Move results to CPU
                    result_cpu = self._move_to_cpu(result)
                    generated_structures.append(result_cpu)
                    
                    log.info(f"Generated structure {i+1}/{num_structures}")
                    
                except Exception as e:
                    log.warning(f"Failed to generate structure {i+1}: {e}")
                    continue
        
        return generated_structures
    
    def generate_from_multiple_embeddings(
        self,
        embeddings: torch.Tensor,
        num_structures_per_embedding: int = 1,
        **generation_kwargs
    ) -> List[List[Dict[str, Any]]]:
        """
        Generate structures from multiple embeddings.
        
        Args:
            embeddings: Tensor of embeddings (batch_size, embedding_dim)
            num_structures_per_embedding: Number of structures per embedding
            **generation_kwargs: Additional generation parameters
            
        Returns:
            List of lists, where each inner list contains structures for one embedding
        """
        all_results = []
        
        for i, embedding in enumerate(embeddings):
            log.info(f"Processing embedding {i+1}/{len(embeddings)}")
            
            structures = self.generate_from_embedding(
                embedding=embedding,
                num_structures=num_structures_per_embedding,
                **generation_kwargs
            )
            
            all_results.append(structures)
        
        return all_results
    
    def _move_to_cpu(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Move tensor data to CPU for serialization."""
        result = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.cpu().numpy().tolist()
            elif isinstance(value, dict):
                result[key] = self._move_to_cpu(value)
            else:
                result[key] = value
        return result


def load_embeddings(
    embedding_path: Union[str, Path],
    embedding_format: str = "auto"
) -> torch.Tensor:
    """
    Load embeddings from various file formats.
    
    Args:
        embedding_path: Path to embedding file
        embedding_format: Format of embeddings ("auto", "pt", "npy", "json")
        
    Returns:
        Loaded embeddings as torch tensor
    """
    embedding_path = Path(embedding_path)
    
    if embedding_format == "auto":
        embedding_format = embedding_path.suffix[1:]  # Remove the dot
    
    log.info(f"Loading embeddings from {embedding_path} (format: {embedding_format})")
    
    if embedding_format == "pt":
        embeddings = torch.load(embedding_path, map_location="cpu")
    elif embedding_format == "npy":
        embeddings = torch.from_numpy(np.load(embedding_path)).float()
    elif embedding_format == "json":
        with open(embedding_path, 'r') as f:
            data = json.load(f)
        embeddings = torch.tensor(data, dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported embedding format: {embedding_format}")
    
    log.info(f"Loaded embeddings with shape: {embeddings.shape}")
    return embeddings


def save_results(
    results: List[Dict[str, Any]],
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """Save generation results with metadata."""
    output_data = {
        "metadata": metadata or {},
        "generated_structures": results,
        "num_structures": len(results)
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    log.info(f"Results saved to {output_path}")


def main():
    """Command-line interface for embedding-based generation."""
    parser = argparse.ArgumentParser(
        description="Generate crystal structures from embeddings"
    )
    
    # Model and embedding arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained MatterMake model checkpoint"
    )
    parser.add_argument(
        "--embedding_file",
        type=str,
        required=True,
        help="Path to embedding file (.pt, .npy, or .json)"
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        help="Embedding dimensionality (auto-detected if not provided)"
    )
    parser.add_argument(
        "--embedding_type",
        type=str,
        choices=["pxrd", "dos", "structure", "composition", "generic"],
        default="generic",
        help="Type of embedding (for metadata)"
    )
    parser.add_argument(
        "--embedding_format",
        type=str,
        choices=["auto", "pt", "npy", "json"],
        default="auto",
        help="Format of embedding file"
    )
    
    # Generation parameters
    parser.add_argument(
        "--num_structures",
        type=int,
        default=5,
        help="Number of structures to generate"
    )
    parser.add_argument(
        "--max_atoms",
        type=int,
        default=50,
        help="Maximum atoms per structure"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--sg_sampling_mode",
        type=str,
        default="sample",
        help="Space group sampling mode"
    )
    parser.add_argument(
        "--lattice_sampling_mode",
        type=str,
        default="sample",
        help="Lattice sampling mode"
    )
    parser.add_argument(
        "--atom_sampling_mode",
        type=str,
        default="sample",
        help="Atom type sampling mode"
    )
    parser.add_argument(
        "--coord_sampling_mode",
        type=str,
        default="sample",
        help="Coordinate sampling mode"
    )
    
    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default="embedding_generated_structures.json",
        help="Output file for generated structures"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for inference"
    )
    
    # Embedding encoder configuration
    parser.add_argument(
        "--encoder_layers",
        type=int,
        default=3,
        help="Number of layers in embedding encoder"
    )
    parser.add_argument(
        "--encoder_dropout",
        type=float,
        default=0.1,
        help="Dropout rate in embedding encoder"
    )
    
    args = parser.parse_args()
    
    # Load embeddings
    embeddings = load_embeddings(args.embedding_file, args.embedding_format)
    
    # Auto-detect embedding dimension if not provided
    if args.embedding_dim is None:
        if embeddings.dim() == 1:
            args.embedding_dim = embeddings.shape[0]
        else:
            args.embedding_dim = embeddings.shape[-1]
        log.info(f"Auto-detected embedding dimension: {args.embedding_dim}")
    
    # Setup embedding encoder configuration
    embedding_encoder_config = {
        "num_layers": args.encoder_layers,
        "dropout": args.encoder_dropout,
        "has_conditioning": False
    }
    
    # Initialize generator
    generator = EmbeddingBasedGenerator(
        checkpoint_path=args.checkpoint,
        embedding_dim=args.embedding_dim,
        device=args.device,
        embedding_encoder_config=embedding_encoder_config
    )
    
    # Handle single vs multiple embeddings
    if embeddings.dim() == 1:
        # Single embedding
        log.info(f"Generating from single {args.embedding_type} embedding")
        structures = generator.generate_from_embedding(
            embedding=embeddings,
            num_structures=args.num_structures,
            max_atoms=args.max_atoms,
            temperature=args.temperature,
            sg_sampling_mode=args.sg_sampling_mode,
            lattice_sampling_mode=args.lattice_sampling_mode,
            atom_discrete_sampling_mode=args.atom_sampling_mode,
            coord_sampling_mode=args.coord_sampling_mode,
        )
        
        # Prepare metadata
        metadata = {
            "embedding_type": args.embedding_type,
            "embedding_file": args.embedding_file,
            "embedding_dim": args.embedding_dim,
            "generation_params": {
                "num_structures": args.num_structures,
                "max_atoms": args.max_atoms,
                "temperature": args.temperature,
                "sampling_modes": {
                    "spacegroup": args.sg_sampling_mode,
                    "lattice": args.lattice_sampling_mode,
                    "atom_type": args.atom_sampling_mode,
                    "coordinates": args.coord_sampling_mode,
                }
            }
        }
        
        save_results(structures, args.output, metadata)
        
    else:
        # Multiple embeddings
        log.info(f"Generating from {len(embeddings)} {args.embedding_type} embeddings")
        
        # Calculate structures per embedding
        structures_per_embedding = max(1, args.num_structures // len(embeddings))
        
        all_structures = generator.generate_from_multiple_embeddings(
            embeddings=embeddings,
            num_structures_per_embedding=structures_per_embedding,
            max_atoms=args.max_atoms,
            temperature=args.temperature,
            sg_sampling_mode=args.sg_sampling_mode,
            lattice_sampling_mode=args.lattice_sampling_mode,
            atom_discrete_sampling_mode=args.atom_sampling_mode,
            coord_sampling_mode=args.coord_sampling_mode,
        )
        
        # Flatten results
        flattened_structures = [struct for batch in all_structures for struct in batch]
        
        metadata = {
            "embedding_type": args.embedding_type,
            "embedding_file": args.embedding_file,
            "embedding_dim": args.embedding_dim,
            "num_input_embeddings": len(embeddings),
            "structures_per_embedding": structures_per_embedding,
            "generation_params": {
                "max_atoms": args.max_atoms,
                "temperature": args.temperature,
                "sampling_modes": {
                    "spacegroup": args.sg_sampling_mode,
                    "lattice": args.lattice_sampling_mode,
                    "atom_type": args.atom_sampling_mode,
                    "coordinates": args.coord_sampling_mode,
                }
            }
        }
        
        save_results(flattened_structures, args.output, metadata)
    
    log.info("Generation completed successfully!")


if __name__ == "__main__":
    main()