"""Training logs: console (rank0) + output_dir/train.log via loguru."""
import os
import sys

import torch.distributed as dist
from loguru import logger


def is_main_for_log():
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def _main_filter(_record):
    return is_main_for_log()


def configure_train_logger(output_dir):
    logger.remove()
    logger.add(sys.stderr, filter=_main_filter, level="INFO", format="{message}")
    if output_dir and is_main_for_log():
        os.makedirs(output_dir, exist_ok=True)
        fp = os.path.join(output_dir, "train.log")
        logger.add(fp, filter=_main_filter, level="INFO", encoding="utf-8", enqueue=True, mode="a",
                    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {message}")
