import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


class CrystalDiffusionProcess:
    def __init__(self, config):
        self.config = config

        if isinstance(config.device, int):
            self.device = f"cuda:{config.device}" if torch.cuda.is_available() else "cpu"
        elif isinstance(config.device, str):
            self.device = config.device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.timesteps = config.timesteps

        if config.beta_schedule == "linear":
            self.betas = torch.linspace(config.beta_start, config.beta_end, config.timesteps, device=self.device)
        elif config.beta_schedule == "cosine":
            steps = config.timesteps + 1
            x = torch.linspace(0, config.timesteps, steps, device=self.device)
            alphas_cumprod = torch.cos(((x / config.timesteps) + config.cosine_s_shift) / (1 + config.cosine_s_shift) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clamp(betas, 0, 0.999)
        else:
            raise ValueError(f"Unknown beta schedule: {config.beta_schedule}")

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = {}
            for key, tensor in x_start.items():
                if key == 'atom_types':
                    if tensor.dtype == torch.long or tensor.dtype == torch.int:
                        num_types = self.config.num_atom_types
                        one_hot = F.one_hot(tensor, num_classes=num_types).float()
                        noise[key] = torch.randn_like(one_hot)
                    else:
                        noise[key] = torch.nn.functional.gumbel_softmax(
                            torch.zeros_like(tensor, dtype=torch.float),
                            tau=1.0, hard=False
                        )
                else:
                    noise[key] = torch.randn_like(tensor)

        x_t = {}
        for key, x0 in x_start.items():
            if key == 'atom_types' and (x0.dtype == torch.long or x0.dtype == torch.int):
                num_types = self.config.num_atom_types
                x0 = F.one_hot(x0, num_classes=num_types).float()

            t_idx = t
            if x0.dim() > 1:
                t_idx = t.reshape(-1, *([1] * (x0.dim() - 1)))

            a = extract(self.sqrt_alphas_cumprod, t_idx, x0.shape)
            b = extract(self.sqrt_one_minus_alphas_cumprod, t_idx, x0.shape)

            if key == 'atom_types':
                x_t[key] = a * x0.float() + b * noise[key]
            else:
                x_t[key] = a * x0 + b * noise[key]

        return x_t, noise

    def p_losses(self, denoise_model, x_start, t, condition=None, noise=None):
        x_noisy, noise_dict = self.q_sample(x_start, t, noise=noise)

        noise_pred = denoise_model(x_noisy, t, condition)

        losses = {}
        total_loss = 0

        for key in x_start.keys():
            if key == 'atom_types':
                target_noise = noise_dict[key]
                pred_noise = noise_pred[key]

                losses[key] = F.mse_loss(pred_noise, target_noise)
            else:
                target_noise = noise_dict[key]
                pred_noise = noise_pred[key]

                losses[key] = F.mse_loss(pred_noise, target_noise)

            weight = getattr(self.config, f"{key}_loss_weight", 1.0)
            total_loss += weight * losses[key]

        losses['total'] = total_loss
        return losses

    @torch.no_grad()
    def p_sample(self, model, x_t, t, condition=None):
        """
        Forward pass of the diffusion model.

        Args:
            model: The model to predict noise
            x_t: Dictionary containing noisy crystal data
            t: Diffusion timesteps
            condition: Optional [B, latent_dim] tensor of latent embeddings for conditioning

        Returns:
            Dictionary with predicted next step
        """
        noise_pred = model(x_t, t, condition)

        x_t_minus_1 = {}

        for key, x_tensor in x_t.items():
            t_idx = t
            if x_tensor.dim() > 1:
                t_idx = t.reshape(-1, *([1] * (x_tensor.dim() - 1)))

            posterior_mean = (
                extract(self.posterior_mean_coef1, t_idx, x_tensor.shape) * x_tensor -
                extract(self.posterior_mean_coef2, t_idx, x_tensor.shape) * noise_pred[key]
            )

            if t[0] > 0:
                noise = torch.randn_like(x_tensor)
                variance = torch.exp(0.5 * extract(
                    self.posterior_log_variance_clipped, t_idx, x_tensor.shape
                ))
                x_t_minus_1[key] = posterior_mean + variance * noise

                if key == 'atom_types' and t[0] <= self.config.discretize_t:
                    if self.config.use_argmax_for_discretization:
                        num_atom_types = self.config.num_atom_types
                        logits = x_t_minus_1[key].reshape(-1, num_atom_types)
                        hard_categorical = torch.zeros_like(logits)
                        hard_categorical.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
                        x_t_minus_1[key] = hard_categorical.reshape(x_tensor.shape)
                    else:
                        temperature = max(1.0 - t[0] / self.config.discretize_t, 0.1)
                        num_atom_types = self.config.num_atom_types
                        logits = x_t_minus_1[key].reshape(-1, num_atom_types) / temperature
                        x_t_minus_1[key] = F.softmax(logits, dim=-1).reshape(x_tensor.shape)
            else:
                x_t_minus_1[key] = posterior_mean

        return x_t_minus_1

    @torch.no_grad()
    def p_sample_loop(self, model, shape, condition=None):
        """
        Generate samples by iteratively denoising from random noise.

        Args:
            model: The noise prediction model
            shape: Tuple of (batch_size, num_atoms) for generation
            condition: Optional conditioning tensor (latent code)

        Returns:
            Dictionary with generated crystal structures
        """
        device = next(model.parameters()).device

        if not isinstance(shape, tuple) or len(shape) != 2:
            raise ValueError(f"Shape should be a tuple of (batch_size, num_atoms), got {shape}")

        batch_size, num_atoms = shape

        if batch_size <= 0 or num_atoms <= 0:
            raise ValueError(f"Invalid shape parameters: batch_size={batch_size}, num_atoms={num_atoms}")

        if condition is not None:
            if not torch.is_tensor(condition):
                try:
                    condition = torch.tensor(condition, device=device, dtype=torch.float)
                except Exception as e:
                    raise TypeError(f"Could not convert condition to tensor: {str(e)}")
            elif condition.device != device:
                condition = condition.to(device)

        x = {
            'atom_types': torch.randn((batch_size, num_atoms, self.config.num_atom_types), device=device),
            'positions': torch.randn((batch_size, num_atoms, 3), device=device),
            'lattice': torch.randn((batch_size, 6), device=device)
        }

        try:
            for i in tqdm(reversed(range(0, self.timesteps)), desc='Sampling', total=self.timesteps):
                t = torch.full((batch_size,), i, device=device, dtype=torch.long)
                x = self.p_sample(model, x, t, condition)
        except Exception as e:
            raise RuntimeError(f"Error during diffusion sampling: {str(e)}")

        final_crystal = self._finalize_crystal(x)
        return final_crystal

    def _finalize_crystal(self, x):
        if 'atom_types' in x:
            atom_types = x['atom_types'].argmax(dim=-1)
            x['atom_types'] = atom_types

        if 'lattice' in x:
            lattice = x['lattice'].clone()

            lattice[..., :3] = torch.abs(lattice[..., :3]) + 0.1

            lattice[..., :3] = torch.clamp(lattice[..., :3], min=1.0, max=50.0)

            lattice[..., 3:] = torch.clamp(lattice[..., 3:], 30.0, 150.0)

            x['lattice'] = lattice

        if 'positions' in x:
            pos = x['positions']
            pos = pos - torch.floor(pos)
            x['positions'] = pos

        return x

    def encode(self, model, x_start, num_encoding_steps=None):
        if num_encoding_steps is None:
            num_encoding_steps = self.timesteps // 2

        t = torch.full((x_start['positions'].shape[0],), num_encoding_steps, device=x_start['positions'].device)
        x_t, _ = self.q_sample(x_start, t)

        return x_t, t

    def decode(self, model, x_t, t, condition=None):
        x = {k: v.clone() for k, v in x_t.items()}

        batch_size = x['positions'].shape[0]
        device = x['positions'].device

        for i in tqdm(reversed(range(0, t[0].item())), desc='Decoding'):
            timestep = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, timestep, condition)

        return self._finalize_crystal(x)

    def interpolate(self, model, x_1, x_2, num_steps=10, condition=None, alpha=None):
        encoding_step = self.timesteps // 2
        x_1_latent, t1 = self.encode(model, x_1, encoding_step)
        x_2_latent, t2 = self.encode(model, x_2, encoding_step)

        if alpha is None:
            alpha = torch.linspace(0, 1, num_steps, device=x_1_latent['positions'].device)

        results = []

        for a in alpha:
            x_t = {}
            for key in x_1_latent.keys():
                x_t[key] = (1 - a) * x_1_latent[key] + a * x_2_latent[key]

            interpolated = self.decode(model, x_t, t1, condition)
            results.append(interpolated)

        return results


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())

    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
