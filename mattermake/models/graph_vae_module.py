import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from mattermake.models.components.graph_vae import QuotientGraphVAE


class GraphVAEModule(LightningModule):
    def __init__(
        self,
        node_feat_dim: int = 103,  # 100 for elements + 3 for fractional coords
        edge_feat_dim: int = 4,  # 1 for distance + 3 for periodicity
        hidden_dim: int = 256,
        latent_dim: int = 128,
        max_nodes: int = 50,
        learning_rate: float = 1e-3,
        beta: float = 0.5,  # KL divergence weight
        train_cell_params: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.beta = beta
        self.learning_rate = learning_rate
        self.train_cell_params = train_cell_params

        self.vae = QuotientGraphVAE(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            max_nodes=max_nodes,
        )

    def forward(self, batch):
        node_features = batch.x
        edge_index = batch.edge_index
        edge_features = batch.edge_attr

        decoded, mu, log_var = self.vae(node_features, edge_index, edge_features)

        return decoded, mu, log_var

    def compute_loss(self, batch, decoded, mu, log_var):
        node_features_true = batch.x
        max_nodes = self.vae.decoder.max_nodes
        num_nodes_to_use = min(batch.num_nodes, max_nodes)
        node_features_true_trunc = node_features_true[:num_nodes_to_use]
        node_features_pred = decoded["node_features"][0, :num_nodes_to_use]
        node_loss = F.mse_loss(node_features_pred, node_features_true_trunc)

        adj_true = torch.zeros(num_nodes_to_use, num_nodes_to_use, device=self.device)

        # Filter edge_index to only include edges between nodes we're considering
        mask = (batch.edge_index[0] < num_nodes_to_use) & (
            batch.edge_index[1] < num_nodes_to_use
        )
        filtered_edge_index = batch.edge_index[:, mask]
        filtered_edge_attr = batch.edge_attr[mask]

        adj_true[filtered_edge_index[0], filtered_edge_index[1]] = 1

        adj_pred_logits = decoded["edge_logits"][
            0, :num_nodes_to_use, :num_nodes_to_use
        ]
        edge_loss = F.binary_cross_entropy_with_logits(adj_pred_logits, adj_true)

        if filtered_edge_index.size(1) > 0:
            edge_features_pred = self.vae.decoder.generate_edge_features(
                filtered_edge_index, decoded["edge_h"]
            )
            edge_feat_loss = F.mse_loss(edge_features_pred, filtered_edge_attr)
        else:
            edge_feat_loss = torch.tensor(0.0, device=self.device)

        # Initialize cell_loss (will be 0 if not training cell_params)
        cell_loss = torch.tensor(0.0, device=self.device)
        cell_params_true_shape = 0
        cell_params_pred_shape = 0

        # Only compute cell loss if train_cell_params is True
        if self.train_cell_params:
            cell_params_true = batch.cell_params
            cell_params_pred = decoded["cell_params"][0]

            cell_params_true_shape = cell_params_true.shape[0]
            cell_params_pred_shape = cell_params_pred.shape[0]

            if cell_params_pred.shape != cell_params_true.shape:
                if cell_params_pred.numel() >= cell_params_true.numel():
                    # Take just what we need (first 6 elements)
                    cell_params_pred = cell_params_pred[: cell_params_true.numel()]
                else:
                    pad_size = cell_params_true.numel() - cell_params_pred.numel()
                    cell_params_pred = torch.cat(
                        [cell_params_pred, torch.zeros(pad_size, device=self.device)]
                    )

            cell_loss = F.mse_loss(cell_params_pred, cell_params_true)

        kl_loss = (
            -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
        )

        # Add a penalty for having too many nodes in input vs. max capacity
        # This encourages the model to handle smaller structures better
        if batch.num_nodes > max_nodes:
            node_count_penalty = torch.log(
                torch.tensor(batch.num_nodes / max_nodes, device=self.device)
            )
        else:
            node_count_penalty = torch.tensor(0.0, device=self.device)

        # Only include cell_loss in recon_loss if train_cell_params is True
        recon_loss = (
            node_loss
            + edge_loss
            + edge_feat_loss
            + (cell_loss if self.train_cell_params else 0.0)
            + 0.1 * node_count_penalty
        )
        current_epoch = self.current_epoch
        total_epochs = self.trainer.max_epochs
        beta_weight = min(
            1.0, current_epoch / (total_epochs * 0.3)
        )  # Ramp up over 30% of training
        total_loss = recon_loss + beta_weight * self.beta * kl_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "node_loss": node_loss,
            "edge_loss": edge_loss,
            "edge_feat_loss": edge_feat_loss,
            "cell_loss": cell_loss,
            "node_count_penalty": node_count_penalty
            if isinstance(node_count_penalty, torch.Tensor)
            else torch.tensor(node_count_penalty),
            "cell_params_true_shape": cell_params_true_shape,
            "cell_params_pred_shape": cell_params_pred_shape,
        }

    def training_step(self, batch, batch_idx):
        decoded, mu, log_var = self(batch)
        loss_dict = self.compute_loss(batch, decoded, mu, log_var)

        self.log(
            "train_loss", loss_dict["loss"], batch_size=batch.num_nodes, sync_dist=True
        )
        self.log(
            "train_recon_loss",
            loss_dict["recon_loss"],
            batch_size=batch.num_nodes,
            sync_dist=True,
        )
        self.log(
            "train_kl_loss",
            loss_dict["kl_loss"],
            batch_size=batch.num_nodes,
            sync_dist=True,
        )

        self.logger.log_metrics(
            {
                "train_cell_params_true_shape": loss_dict.get(
                    "cell_params_true_shape", 6
                ),
                "train_cell_params_pred_shape": loss_dict.get(
                    "cell_params_pred_shape", 6
                ),
            },
            step=self.global_step,
        )

        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        decoded, mu, log_var = self(batch)
        loss_dict = self.compute_loss(batch, decoded, mu, log_var)

        self.log(
            "val_loss", loss_dict["loss"], batch_size=batch.num_nodes, sync_dist=True
        )
        self.log(
            "val_recon_loss",
            loss_dict["recon_loss"],
            batch_size=batch.num_nodes,
            sync_dist=True,
        )
        self.log(
            "val_kl_loss",
            loss_dict["kl_loss"],
            batch_size=batch.num_nodes,
            sync_dist=True,
        )
        self.log(
            "val_node_loss",
            loss_dict["node_loss"],
            batch_size=batch.num_nodes,
            sync_dist=True,
        )
        self.log(
            "val_edge_loss",
            loss_dict["edge_loss"],
            batch_size=batch.num_nodes,
            sync_dist=True,
        )
        self.log(
            "val_edge_feat_loss",
            loss_dict["edge_feat_loss"],
            batch_size=batch.num_nodes,
            sync_dist=True,
        )

        # Only log cell_loss if we're training cell params
        if self.train_cell_params:
            self.log(
                "val_cell_loss",
                loss_dict["cell_loss"],
                batch_size=batch.num_nodes,
                sync_dist=True,
            )

        self.logger.log_metrics(
            {
                "val_cell_params_true_shape": loss_dict.get(
                    "cell_params_true_shape", 6
                ),
                "val_cell_params_pred_shape": loss_dict.get(
                    "cell_params_pred_shape", 6
                ),
            },
            step=self.global_step,
        )

        return loss_dict["loss"]

    def test_step(self, batch, batch_idx):
        decoded, mu, log_var = self(batch)
        loss_dict = self.compute_loss(batch, decoded, mu, log_var)

        self.log("test_loss", loss_dict["loss"])
        self.log("test_recon_loss", loss_dict["recon_loss"])
        self.log("test_kl_loss", loss_dict["kl_loss"])

        return loss_dict["loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def generate(self, num_samples=1):
        """Generate new crystal structures from random points in the latent space"""
        z = torch.randn(num_samples, self.hparams.latent_dim).to(self.device)

        with torch.no_grad():
            decoded = self.vae.decode(z)

        return decoded

    def reconstruct(self, batch):
        """Reconstruct crystal structures from input data"""
        with torch.no_grad():
            z, _, _ = self.vae.encode(batch.x, batch.edge_index, batch.edge_attr)
            decoded = self.vae.decode(z)

        return decoded

    def on_train_start(self):
        """Log the model size when training starts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.logger.log_metrics(
            {
                "model/total_parameters": total_params,
                "model/trainable_parameters": trainable_params,
            }
        )
        print(
            f"Model size: {total_params / 1e6:.2f}M parameters ({trainable_params / 1e6:.2f}M trainable)"
        )
