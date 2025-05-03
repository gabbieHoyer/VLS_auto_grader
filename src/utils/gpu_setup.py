# src/utils/gpu_setup.py
import os
import random
import logging
import builtins
import numpy as np

import torch
import torch.distributed as dist

class GPUSetup:
    @staticmethod
    def setup(distributed: bool = False, seed: int = 42):
        # 1) pull env vars
        rank       = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        # 2) detect how many GPUs you actually have
        available_gpus = torch.cuda.device_count()
        if distributed and available_gpus < 2:
            builtins.print(f"Only {available_gpus} GPU(s) detected - switching to single - process mode.")
            distributed = False
            world_size  = 1
            rank        = 0
            local_rank  = 0

        # 3) always set your threading env vars
        os.environ["OMP_NUM_THREADS"]        = "4"
        os.environ["OPENBLAS_NUM_THREADS"]   = "4"
        os.environ["MKL_NUM_THREADS"]        = "6"
        os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
        os.environ["NUMEXPR_NUM_THREADS"]    = "6"

        # 4) pick a backend and init URL
        backend     = "nccl" if torch.cuda.is_available() else "gloo"
        master_addr = os.getenv("MASTER_ADDR", "localhost")
        master_port = os.getenv("MASTER_PORT", "5675")
        init_url    = os.getenv("DIST_URL", f"tcp://{master_addr}:{master_port}")

        # 5) *always* initialize a process group of size=world_size
        dist.init_process_group(
            backend=backend,
            init_method=init_url,
            rank=rank,
            world_size=world_size,
        )
        dist.barrier()

        # 6) silence non‐master prints
        import builtins as __builtin__
        builtin_print = __builtin__.print
        def print(*args, force=False, **kwargs):
            if rank == 0 or force:
                builtin_print(*args, **kwargs)
        __builtin__.print = print

        print(f"Process group initialized: rank {rank}/{world_size}, local_rank={local_rank}", force=True)

        # 7) set device & seeds
        device_str = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(local_rank) if torch.cuda.is_available() else None
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.enabled     = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

        # 8) store for helpers
        GPUSetup._rank       = rank
        GPUSetup._world_size = world_size
        GPUSetup._local_rank = local_rank

    @staticmethod
    def is_distributed():
        return dist.is_initialized()

    @staticmethod
    def get_rank():
        return dist.get_rank() if dist.is_initialized() else 0

    @staticmethod
    def get_world_size():
        return dist.get_world_size() if dist.is_initialized() else 1

    @staticmethod
    def get_local_rank():
        return GPUSetup._local_rank

    @staticmethod
    def is_main_process():
        return GPUSetup.get_rank() == 0

    @staticmethod
    def save_on_master(*args, **kwargs):
        if GPUSetup.is_main_process():
            torch.save(*args, **kwargs)

    @staticmethod
    def reduce_tensor(tensor, average=True):
        """
        Reduces the tensor across all processes in distributed training.
        """
        if not GPUSetup.is_distributed():
            return tensor
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        if average:
            rt /= GPUSetup.get_world_size()
        return rt

    @staticmethod
    def cleanup():
        if GPUSetup.is_distributed():
            dist.destroy_process_group()
        torch.cuda.empty_cache()  # Helps in releasing unreferenced memory immediately

