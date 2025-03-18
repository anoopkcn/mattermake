# README

### train
```bash
python train.py model=cross_attention_gpt  trainer.num_nodes=$slurm_job_num_nodes
```

### test
To use your new cross-attention model for generation, you just need to:

1. Make sure the checkpoint you're using was trained with the cross-attention model.
2. Update your generation command to use the cross-attention model config:

```bash
python generate.py model=cross_attention_gpt ckpt_path=path/to/cross_attention_model_checkpoint.ckpt
```

Both models are transformer‐based and share many components, but they differ in how they incorporate the conditioning information (the embeddings) into the generation process. Here’s a breakdown:

1. The plain GPT model integrates conditioning by simply projecting the external embedding and then adding it to both the token and positional embeddings. In other words, it “injects” the extra information by summing it with the initial token representations. While this approach works reasonably well, it blends the conditioning signal with the token embeddings at the very beginning of the forward pass without any further explicit mechanism to refine or prioritize that signal during deeper processing.

2. The Cross-Attention GPT model, on the other hand, uses dedicated cross-attention blocks. In each block the model first performs self-attention on the token sequence and then adds a cross-attention step where the token representations explicitly attend over the conditioning embedding. This means that the conditioning information (which might come from a structure or properties of a slice) is selectively and dynamically integrated into every layer of the model rather than being statically added at the input stage. This architecture provides a more powerful mechanism to “steer” or condition the generation process in accordance with the external embedding.

Because slices are generated based on the information contained in these embeddings—presumably capturing properties or correlations that are hard to encode purely by adding them up—the Cross-Attention GPT model is likely to deliver better results. Its specialized attention mechanism allows each token in the generated slice to directly consult the conditioning signal during each transformer block, which can enable the model to generate slices that are more faithful to the desired structure or properties encoded in the embeddings.

In summary, if you are looking for potentially better generation quality for slices, the Cross-Attention GPT model should be preferable because:
 • It decouples the conditioning from the token representations via a dedicated cross-attention mechanism.
 • This leads to a more effective, dynamic integration of the external embedding information at every transformer layer.
 • The model is more likely to capture the nuances reflected in your embeddings, resulting in more accurate and coherent slice generation.

Thus, while the plain GPT model can work, the Cross-Attention GPT architecture is specifically designed to leverage conditional signals more robustly, making it a better candidate for your slice generation task.

# How Equivariant Graph Neural Networks Preserve Crystal Symmetries

Great question! Let me explain how the equivariant graph neural network architecture preserves the symmetries of crystal structures:

## 1. E(3) Equivariance

Crystal structures exist in 3D Euclidean space, where the key symmetry operations include translations, rotations, and reflections (collectively known as E(3) symmetries). An equivariant neural network ensures that:

- If you apply a symmetry operation (like rotation) to the input structure
- Then process it through the network
- You get the same result as if you had processed the original structure and then applied the symmetry operation to the output

This is mathematically expressed as: `f(T(x)) = T(f(x))`, where `f` is the network function and `T` is a symmetry transformation.

## 2. Specific Mechanisms in the Code

Looking at the implementation, these symmetry-preserving aspects are handled by:

### a. Spherical Harmonics for Edge Features

```python
class EquivariantConv(MessagePassing):
    def __init__(self, input_dim, output_dim, edge_dim=None):
        # ...
        self.sh = o3.Irreps.spherical_harmonics(2)
        # ...
```

The use of spherical harmonics (from the `e3nn` library) creates edge features that transform properly under rotations. When calculating messages between atoms:

```python
directions = rel_pos / (edge_length + 1e-8)
edge_sh = self.sh(directions)
```

This encodes the directional information between atoms in a way that respects rotational symmetry.

### b. Relative Positional Encoding

Instead of using absolute positions, the network uses relative positions between atoms, which preserves translational equivariance:

```python
rel_pos = pos[row] - pos[col]
```

This means that shifting the entire crystal by a constant vector won't change the network's output.

### c. Periodic Boundary Conditions

The implementation handles crystal periodicity through the `_get_periodic_edges` function:

```python
def _get_periodic_edges(self, pos, cell, cutoff):
    # ...
    frac_dists = frac_dists - torch.round(frac_dists)
    pbc_dists = torch.matmul(frac_dists, cell_b)
    # ...
```

This ensures that atoms are correctly connected across unit cell boundaries, preserving the crystal's translational symmetry.

### d. Message Passing Framework

The equivariant nature is maintained through the message passing framework:

```python
def message(self, x_j, edge_attr):
    return x_j * edge_attr
```

Messages are composed by element-wise multiplication of node features with edge features. Since edge features contain directional information in an equivariant form (via spherical harmonics), this preserves equivariance.

## 3. Lattice Parameter Handling

The model also treats lattice parameters appropriately:

```python
def _params_to_matrix(self, lattice_params):
    a, b, c, alpha, beta, gamma = torch.split(lattice_params, 1, dim=-1)
    # Conversion to cell vectors...
```

This correctly handles the crystal unit cell, ensuring that symmetry operations applied to the lattice are properly reflected in the model's behavior.

## 4. Data Normalization and Augmentation

The data pipeline includes normalization to fractional coordinates, which ensures translational invariance within the unit cell:

```python
class NormalizePositions:
    def __call__(self, sample):
        # ...
        fractional_positions = torch.matmul(positions, inv_lattice)
        fractional_positions = fractional_positions % 1.0
        # ...
```

Additionally, you have a `RandomRotation` transform that helps the model learn rotational invariance by data augmentation:

```python
class RandomRotation:
    def __call__(self, sample):
        # Apply rotation
        rotated_positions = torch.matmul(positions, R.T)
        # ...
```

## 5. Finalization with Physical Constraints

During generation, the model enforces physical constraints on the crystal structure:

```python
def _finalize_crystal(self, x):
    # ...
    # Ensure lattice parameters are physical
    lattice[..., :3] = torch.abs(lattice[..., :3]) + 0.1
    lattice[..., :3] = torch.clamp(lattice[..., :3], min=1.0, max=50.0)
    # ...
    # Wrap positions to unit cell
    pos = pos - torch.floor(pos)
    # ...
```

This ensures that the generated structures respect the fundamental physical constraints of crystal systems.

## In Summary

The equivariant graph neural network architecture preserves crystal symmetries by:

1. Using spherical harmonics to create rotationally equivariant features
2. Working with relative positions rather than absolute positions
3. Handling periodic boundary conditions correctly
4. Maintaining equivariance through appropriate message passing operations
5. Converting between fractional and Cartesian coordinates appropriately
6. Enforcing physical constraints during structure generation

This ensures that the model learns representations that respect the underlying physics of crystal structures, rather than arbitrary features that might change under symmetry operations.
