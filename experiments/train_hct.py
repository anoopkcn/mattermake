import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from mattermake.data.hct_datamodule import HCTDataModule
from mattermake.models.hct import HierarchicalCrystalTransformer
import os


@hydra.main(
    config_path="mattermake/configs", config_name="hct_config", version_base="1.3"
)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Set project root for relative paths if needed
    if "PROJECT_ROOT" not in os.environ:
        os.environ["PROJECT_ROOT"] = os.getcwd()

    # Instantiate DataModule
    datamodule = HCTDataModule(**cfg.datamodule)

    # Instantiate Model
    model = HierarchicalCrystalTransformer(**cfg.model)

    # Callbacks (optional, can be extended)
    callbacks = []

    # Logger (optional, can be extended)
    logger = None

    # Trainer
    trainer = pl.Trainer(**cfg.trainer, callbacks=callbacks, logger=logger)

    # Fit
    trainer.fit(model, datamodule=datamodule)

    # Optionally test after training
    if cfg.get("test_after_train", False):
        trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
