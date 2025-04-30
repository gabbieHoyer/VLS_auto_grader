import logging
import torch
import os

from functools import wraps

# from . import gpu_setup as GPUSetup
from . import GPUSetup

logger = logging.getLogger(__name__)

def setup_logging(config_level, logger):
    level_mapping = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
    }
    logging_level = level_mapping.get(config_level, logging.INFO)
    rank = GPUSetup.get_rank()  # Now uses GPUSetup
    logging.basicConfig(
        level=logging_level,
        format=f'%(asctime)s - %(levelname)s - Rank {rank} - %(filename)s:%(lineno)d - %(funcName)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger.info("PyTorch version: %s", torch.__version__)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    logger.info("Number of GPUs available: %s", torch.cuda.device_count())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            logger.info("GPU %s: %s", i, torch.cuda.get_device_name(i))
        logger.info("CUDA_VISIBLE_DEVICES: %s", os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set'))
    return logger



def setup_logger(name, log_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # File handler
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, f'{name}.log'))
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger



# -------- DECORATOR FOR MAIN PROCESS ONLY FUNCTIONALITY -------- #
def main_process_only(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if GPUSetup.is_main_process():
            return func(*args, **kwargs)
    return wrapper

@main_process_only
def log_info(message):
    logger.info(message, stacklevel=2)

@main_process_only
def wandb_log(data):
    import wandb  
    wandb.log(data)