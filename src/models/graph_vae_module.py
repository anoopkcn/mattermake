import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from src.models.components.graph_vae import QuotientGraphVAE


class GraphVAEModule(LightningModule):
    def __init__(
        self,
        node_feat_dim: int = 103,  # 100 for elements + 3 for fractional coords
        edge_feat_dim: int = 4,  # 1 for distance + 3 for periodicity
        hidden_dim: int = 256,
        latent_dim: int = 128,
        max_nodes: int = 50,
        learning_rate: float = 1e-3,
        beta: float = 1.0,  # KL divergence weight
    ):
        super().__init__()
        self.save_hyperparameters()

        self.beta = beta
        self.learning_rate = learning_rate

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
        # Node feature reconstruction loss
        node_features_true = batch.x
        node_features_pred = decoded["node_features"][:, : batch.num_nodes]
        node_loss = F.mse_loss(node_features_pred, node_features_true)

        # Edge existence loss (binary cross entropy)
        # Create target adjacency matrix from edge_index
        adj_true = torch.zeros(batch.num_nodes, batch.num_nodes, device=self.device)
        adj_true[batch.edge_index[0], batch.edge_index[1]] = 1

        # Get predicted adjacency
        adj_pred_logits = decoded["edge_logits"][
            :, : batch.num_nodes, : batch.num_nodes
        ]
        edge_loss = F.binary_cross_entropy_with_logits(adj_pred_logits, adj_true)

        # Edge feature reconstruction loss for existing edges
        edge_features_true = batch.edge_attr
        # We would generate features for predicted edges, but for simplicity,
        # we'll just use the ground truth edge indices
        edge_features_pred = self.vae.decoder.generate_edge_features(
            batch.edge_index, decoded["edge_h"]
        )
        edge_feat_loss = F.mse_loss(edge_features_pred, edge_features_true)

        cell_params_true = batch.cell_params
        cell_params_pred = decoded["cell_params"]
        cell_loss = F.mse_loss(cell_params_pred, cell_params_true)

        kl_loss = (
            -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
        )

        recon_loss = node_loss + edge_loss + edge_feat_loss + cell_loss
        total_loss = recon_loss + self.beta * kl_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "node_loss": node_loss,
            "edge_loss": edge_loss,
            "edge_feat_loss": edge_feat_loss,
            "cell_loss": cell_loss,
        }

    def training_step(self, batch, batch_idx):
        decoded, mu, log_var = self(batch)
        loss_dict = self.compute_loss(batch, decoded, mu, log_var)

        self.log("train_loss", loss_dict["loss"])
        self.log("train_recon_loss", loss_dict["recon_loss"])
        self.log("train_kl_loss", loss_dict["kl_loss"])

        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        decoded, mu, log_var = self(batch)

        loss_dict = self.compute_loss(batch, decoded, mu, log_var)

        self.log("val_loss", loss_dict["loss"])
        self.log("val_recon_loss", loss_dict["recon_loss"])
        self.log("val_kl_loss", loss_dict["kl_loss"])
        self.log("val_node_loss", loss_dict["node_loss"])
        self.log("val_edge_loss", loss_dict["edge_loss"])
        self.log("val_edge_feat_loss", loss_dict["edge_feat_loss"])
        self.log("val_cell_loss", loss_dict["cell_loss"])

        return loss_dict["loss"]

    def test_step(self, batch, batch_idx):
        # Forward pass
        decoded, mu, log_var = self(batch)

        # Compute loss
        loss_dict = self.compute_loss(batch, decoded, mu, log_var)

        # Log metrics
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
