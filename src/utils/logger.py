# src/utils/logger.py
import os
import logging
import torch

from . import GPUSetup

def setup_logging(config_level, log_file=None):
    """
    Configure the *root* logger:
      • stamps every LogRecord with record.rank
      • adds a console handler
      • optional file handler
    """
    # map level
    LEVELS = {
      "DEBUG": logging.DEBUG,
      "INFO":  logging.INFO,
      "WARNING": logging.WARNING,
      "ERROR": logging.ERROR,
      "CRITICAL": logging.CRITICAL,
    }
    lvl = LEVELS.get(config_level, logging.INFO)

    # inject rank into every record
    old_factory = logging.getLogRecordFactory()
    def record_factory(*args, **kwargs):
        rec = old_factory(*args, **kwargs)
        rec.rank = GPUSetup.get_rank()
        return rec
    logging.setLogRecordFactory(record_factory)

    root = logging.getLogger()
    root.setLevel(lvl)
    # clear old handlers
    for h in root.handlers[:]:
        root.removeHandler(h)

    fmt = logging.Formatter(
      "%(asctime)s - %(levelname)s - Rank %(rank)d - "
      "%(filename)s:%(lineno)d - %(funcName)s - %(message)s",
      datefmt="%Y-%m-%d %H:%M:%S"
    )

    # console
    ch = logging.StreamHandler()
    ch.setLevel(lvl)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # file
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a")
        fh.setLevel(lvl)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    # one‐time dumps
    root.info("PyTorch version: %s", torch.__version__)
    root.info("CUDA available: %s", torch.cuda.is_available())
    for i in range(torch.cuda.device_count()):
        root.info(" GPU %d: %s", i, torch.cuda.get_device_name(i))
    root.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES","unset"))

    return root


def main_process_only(func):
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        if GPUSetup.is_main_process():
            return func(*args, **kwargs)
    return wrapper

@main_process_only
def log_info(*args, **kwargs):
    logging.getLogger().info(*args, **kwargs)

@main_process_only
def wandb_log(data):
    import wandb
    wandb.log(data)



