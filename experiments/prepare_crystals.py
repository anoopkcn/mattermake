import hydra
import rootutils
from omegaconf import DictConfig
from lightning.pytorch import seed_everything

from mattermake.data.crystal_prepare import prepare_structure_data
from mattermake.utils.pylogger import get_pylogger
from mattermake.utils.utils import extras
from mattermake.utils.distributed_init import (
    configure_pytorch,
    init_distributed_mode,
    log_distributed_settings,
    patch_lightning_slurm_master_addr,
)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

log = get_pylogger(__name__)


@hydra.main(
    version_base="1.3",
    config_path="../configs",
    config_name="crystals_prepare",
)
def main(cfg: DictConfig) -> None:
    """Prepare data for training.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
    """
    extras(cfg)

    log.info("Starting data preparation...")
    prepare_structure_data(
        input_path=cfg.input_path,
        output_dir=cfg.output_dir,
        train_ratio=cfg.train_ratio,
        data_limit=cfg.data_limit,
        seed=cfg.seed,
    )
    log.info("Data preparation completed!")


if __name__ == "__main__":
    seed_everything(42)
    patch_lightning_slurm_master_addr()
    init_distributed_mode(port=12354)
    log_distributed_settings(log)
    configure_pytorch(log)
    main()
