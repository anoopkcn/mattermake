import math
import torch
from lightning.pytorch import LightningModule

from models.components.gpt_model import GPT, GPTConfig
from utils.vocab import decode_slice, stoi
from utils import pylogger


class GPTModule(LightningModule):
    def __init__(
        self,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        block_size: int = 1024,
        bias: bool = True,
        vocab_size: int = 94,  # Adjust based on your vocab size
        dropout: float = 0.0,
        embedding_dim: int = 256,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-1,
        betas: tuple = (0.9, 0.95),
        warmup_iters: int = 150,
        lr_decay_iters: int = 600000,
        min_lr: float = 3e-5,
        decay_lr: bool = True,
    ):
        super().__init__()
        self.custom_logger = pylogger.get_pylogger(__name__)
        self.save_hyperparameters()

        # Initialize model
        model_args = dict(
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            block_size=block_size,
            bias=bias,
            vocab_size=vocab_size,
            dropout=dropout,
            embedding_dim=embedding_dim,
        )
        config = GPTConfig(**model_args)
        self.model = GPT(config)

    def forward(self, input_ids, embeddings=None):
        return self.model(input_ids, embeddings=embeddings)

    def training_step(self, batch, batch_idx):
        embeddings = batch["embeddings"]
        input_ids = batch["input_ids"]
        targets = batch["target_ids"]

        logits, loss = self.model(input_ids, targets, embeddings=embeddings)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        embeddings = batch["embeddings"]
        input_ids = batch["input_ids"]
        targets = batch["target_ids"]

        logits, loss = self.model(input_ids, targets, embeddings=embeddings)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        # TODO: This can be handled using callback
        """Log epoch summary at the end of validation."""
        train_loss = self.trainer.callback_metrics.get("train_loss", float("nan"))
        val_loss = self.trainer.callback_metrics.get("val_loss", float("nan"))

        # Get current learning rate (if using a learning rate scheduler)
        current_lr = (
            self.optimizers().param_groups[0]["lr"] if self.trainer.optimizers else None
        )

        # Create and log the summary
        summary = f"\n=== Epoch {self.current_epoch} Summary ===\n"
        summary += f"Train Loss: {train_loss:.4f}\n"
        summary += f"Val Loss: {val_loss:.4f}\n"
        if current_lr is not None:
            summary += f"Learning Rate: {current_lr:.2e}\n"
        summary += "=" * 25

        # Use the existing logger
        self.custom_logger.info(summary)

    def configure_optimizers(self):
        # Filter params that require grad
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        # Create optim groups - weight decay for 2D params, no decay for others
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.hparams.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.hparams.learning_rate, betas=self.hparams.betas
        )

        if not self.hparams.decay_lr:
            return optimizer

        # Create learning rate scheduler with explicit ordering
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=self.get_lr_schedule
        )

        # Return both optimizer and scheduler without the additional configuration
        # which makes PyTorch Lightning handle them differently
        return [optimizer], [scheduler]

    def get_lr_schedule(self, iter_num):
        # Linear warmup followed by cosine decay
        warmup_iters = self.hparams.warmup_iters
        lr_decay_iters = self.hparams.lr_decay_iters
        min_lr = self.hparams.min_lr
        learning_rate = self.hparams.learning_rate

        # Linear warmup
        if iter_num < warmup_iters:
            return float(iter_num) / float(max(1, warmup_iters))

        # Cosine decay
        if iter_num > lr_decay_iters:
            return min_lr / learning_rate

        decay_ratio = float(iter_num - warmup_iters) / float(
            max(1, lr_decay_iters - warmup_iters)
        )
        coeff = 0.5 * (1.0 + torch.cos(torch.tensor(math.pi * decay_ratio))).item()
        return (min_lr + coeff * (learning_rate - min_lr)) / learning_rate

    def generate(self, embeddings, max_new_tokens=100, temperature=0.7, top_k=40):
        """Generate slices from embeddings"""
        self.model.eval()
        embedding_tensor = embeddings.to(self.device)

        # Start with <START> token
        start_token_id = stoi["<START>"]
        start_tokens = torch.tensor(
            [[start_token_id]], dtype=torch.long, device=self.device
        )

        with torch.no_grad():
            tokens = self.model.generate(
                idx=start_tokens,
                embeddings=embedding_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )

        generated_slices = [
            decode_slice(tokens[i].tolist()) for i in range(tokens.size(0))
        ]
        return generated_slices

    def batch_generate(self, embeddings, max_new_tokens=100, temperature=0.7, top_k=40):
        """Generate multiple slices from batched embeddings"""
        self.model.eval()
        batch_size = embeddings.shape[0]

        # Start with <START> token for all samples in batch
        start_token_id = stoi["<START>"]
        start_tokens = torch.tensor(
            [[start_token_id] for _ in range(batch_size)],
            dtype=torch.long,
            device=self.device,
        )

        with torch.no_grad():
            tokens = self.model.generate(
                idx=start_tokens,
                embeddings=embeddings,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )

        generated_slices = [
            decode_slice(tokens[i].tolist()) for i in range(tokens.size(0))
        ]
        return generated_slices
