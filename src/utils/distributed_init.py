import os
import re

import torch
from lightning.pytorch.plugins.environments import SLURMEnvironment


def get_first_node():
    """Return the first node we can find in the Slurm node list."""
    nodelist = os.getenv("SLURM_JOB_NODELIST")

    bracket_re = re.compile(r"(.*?)\[(.*?)\]")
    dash_re = re.compile("(.*?)-")
    comma_re = re.compile("(.*?),")

    bracket_result = bracket_re.match(nodelist)

    if bracket_result:
        node = bracket_result[1]
        indices = bracket_result[2]

        comma_result = comma_re.match(indices)
        if comma_result:
            indices = comma_result[1]

        dash_result = dash_re.match(indices)
        first_index = dash_result[1] if dash_result else indices

        return node + first_index

    comma_result = comma_re.match(nodelist)
    if comma_result:
        return comma_result[1]

    return nodelist


def init_distributed_mode(port: int = 12354):
    """Initialize some environment variables for PyTorch Distributed
    using Slurm.
    """
    # The number of total processes started by Slurm.
    os.environ["WORLD_SIZE"] = os.getenv("SLURM_NTASKS")
    # Index of the current process.
    os.environ["RANK"] = os.getenv("SLURM_PROCID")
    # Index of the current process on this node only.
    os.environ["LOCAL_RANK"] = os.getenv("SLURM_LOCALID")

    master_addr = get_first_node()
    systemname = os.getenv("SYSTEMNAME", "")
    # Need to append "i" on Jülich machines to connect across InfiniBand cells.
    if systemname in ["juwels", "juwelsbooster", "jureca"]:
        master_addr = master_addr + "i"
    os.environ["MASTER_ADDR"] = master_addr

    # An arbitrary free port on node 0.
    os.environ["MASTER_PORT"] = str(port)


def log_distributed_settings(logger):
    logger.info(f"MASTER_ADDR={os.getenv('MASTER_ADDR')}")
    logger.info(f"MASTER_PORT={os.getenv('MASTER_PORT')}")
    logger.info(f"WORLD_SIZE={os.getenv('WORLD_SIZE')}")
    logger.info(f"RANK={os.getenv('RANK')}")
    logger.info(f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}")


def configure_pytorch(logger):
    torch.set_float32_matmul_precision("medium")

    try:
        # set PYTORCH_ALLOC_CONF to avoid memory fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    except Exception:
        logger.warning("Your PyTorch version does not support PYTORCH_CUDA_ALLOC_CONF")


def is_running_on_juwels() -> bool:
    return os.getenv("SYSTEMNAME", "") in ["juwelsbooster", "juwels", "jurecadc"]


def patch_lightning_slurm_master_addr():
    if not is_running_on_juwels():
        return

    old_resolver = SLURMEnvironment.resolve_root_node_address

    def new_resolver(nodes):
        # Append an i" for communication over InfiniBand.
        return old_resolver(nodes) + "i"

    SLURMEnvironment.__old_resolve_root_node_address = old_resolver
    SLURMEnvironment.resolve_root_node_address = new_resolver
