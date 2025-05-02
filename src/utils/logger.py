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




# import logging
# import torch
# import os
# from functools import wraps
# from . import GPUSetup

# logger = logging.getLogger(__name__)


# def setup_logging(config_level, log_file=None):
#     """
#     Configure the root logger with:
#       * a custom LogRecordFactory that always sets record.rank
#       * a console handler
#       * an optional file handler
#     """
#     # 1) Map text level to logging constant
#     level_map = {
#         'DEBUG':    logging.DEBUG,
#         'INFO':     logging.INFO,
#         'WARNING':  logging.WARNING,
#         'ERROR':    logging.ERROR,
#         'CRITICAL': logging.CRITICAL,
#     }
#     lvl = level_map.get(config_level, logging.INFO)

#     # 2) Install a LogRecordFactory that stamps every record with .rank
#     old_factory = logging.getLogRecordFactory()
#     def record_factory(*args, **kwargs):
#         record = old_factory(*args, **kwargs)
#         # Always inject the distributed rank
#         record.rank = GPUSetup.get_rank()
#         return record
#     logging.setLogRecordFactory(record_factory)

#     # 3) Grab the root logger, clear old handlers, set level
#     root = logging.getLogger()
#     root.setLevel(lvl)
#     for h in root.handlers[:]:
#         root.removeHandler(h)

#     # 4) Create a formatter that refers to %(rank)d
#     fmt = logging.Formatter(
#         '%(asctime)s - %(levelname)s - Rank %(rank)d - %(filename)s:%(lineno)d - %(funcName)s - %(message)s',
#         datefmt='%Y-%m-%d %H:%M:%S'
#     )

#     # 5) Console handler
#     ch = logging.StreamHandler()
#     ch.setLevel(lvl)
#     ch.setFormatter(fmt)
#     root.addHandler(ch)

#     # 6) Optional file handler
#     if log_file:
#         os.makedirs(os.path.dirname(log_file), exist_ok=True)
#         fh = logging.FileHandler(log_file, mode='a')
#         fh.setLevel(lvl)
#         fh.setFormatter(fmt)
#         root.addHandler(fh)

#     # 7) One-time environment dump
#     root.info("PyTorch version: %s", torch.__version__)
#     root.info("CUDA available: %s", torch.cuda.is_available())
#     for i in range(torch.cuda.device_count()):
#         root.info(" GPU %d: %s", i, torch.cuda.get_device_name(i))
#     root.info("CUDA_VISIBLE_DEVICES=%s", os.environ.get('CUDA_VISIBLE_DEVICES','unset'))

#     return root



# # -------- DECORATOR FOR MAIN PROCESS ONLY FUNCTIONALITY -------- #
# def main_process_only(func):
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         if GPUSetup.is_main_process():
#             return func(*args, **kwargs)
#     return wrapper

# @main_process_only
# def log_info(message):
#     logger.info(message, stacklevel=2)

# @main_process_only
# def wandb_log(data):
#     import wandb  
#     wandb.log(data)



# def setup_logger(name, log_dir):
#     logger = logging.getLogger(name)
#     logger.setLevel(logging.INFO)
    
#     # File handler
#     os.makedirs(log_dir, exist_ok=True)
#     file_handler = logging.FileHandler(os.path.join(log_dir, f'{name}.log'))
#     file_handler.setLevel(logging.INFO)
    
#     # Console handler
#     console_handler = logging.StreamHandler()
#     console_handler.setLevel(logging.INFO)
    
#     # Formatter
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     file_handler.setFormatter(formatter)
#     console_handler.setFormatter(formatter)
    
#     # Add handlers
#     logger.addHandler(file_handler)
#     logger.addHandler(console_handler)
    
#     return logger



# # def setup_logging(config_level, logger, log_file=None):
# #     level_mapping = {
# #         'DEBUG': logging.DEBUG,
# #         'INFO': logging.INFO,
# #         'WARNING': logging.WARNING,
# #         'ERROR': logging.ERROR,
# #         'CRITICAL': logging.CRITICAL,
# #     }
# #     logging_level = level_mapping.get(config_level, logging.INFO)
# #     rank = GPUSetup.get_rank()  # Now uses GPUSetup
# #     logging.basicConfig(
# #         level=logging_level,
# #         format=f'%(asctime)s - %(levelname)s - Rank {rank} - %(filename)s:%(lineno)d - %(funcName)s - %(message)s',
# #         datefmt='%Y-%m-%d %H:%M:%S'
# #     )
# #     logger.info("PyTorch version: %s", torch.__version__)
# #     logger.info("CUDA available: %s", torch.cuda.is_available())
# #     logger.info("Number of GPUs available: %s", torch.cuda.device_count())
# #     if torch.cuda.is_available():
# #         for i in range(torch.cuda.device_count()):
# #             logger.info("GPU %s: %s", i, torch.cuda.get_device_name(i))
# #         logger.info("CUDA_VISIBLE_DEVICES: %s", os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set'))
# #     return logger

