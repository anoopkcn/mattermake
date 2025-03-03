import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback

class EpochSummaryCallback(Callback):
    """Callback that prints a summary after each epoch."""

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called when the validation epoch ends."""
        if not trainer.sanity_checking:
            epoch = trainer.current_epoch
            train_loss = trainer.callback_metrics.get("train_loss", float('nan'))
            val_loss = trainer.callback_metrics.get("val_loss", float('nan'))

            # Get current learning rate
            optimizer = trainer.optimizers[0]
            current_lr = optimizer.param_groups[0]['lr']

            print(f"\n=== Epoch {epoch} Summary ===")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {current_lr:.2e}")
            print("=" * 25)
