from typing import Any, Dict, Optional, Union, cast

from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf

from mattermake.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    if cfg is None or not isinstance(cfg, dict):
        log.warning("Config is not a dictionary! Skipping hyperparameter logging...")
        return

    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    if "model" in cfg:
        hparams["model"] = cfg["model"]
    else:
        log.warning("Model config not found in cfg")

    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    if "data" in cfg:
        hparams["data"] = cfg["data"]
    if "trainer" in cfg:
        hparams["trainer"] = cfg["trainer"]

    for key in ["callbacks", "extras", "task_name", "tags", "ckpt_path", "seed"]:
        if key in cfg:
            hparams[key] = cfg[key]

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
