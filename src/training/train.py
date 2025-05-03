# src/training/train.py
import os
import argparse
import logging
import pandas as pd
from datetime import datetime

import torch
from torch.utils.data import DataLoader, DistributedSampler

from src.utils import (
    GPUSetup, setup_logging, load_config,
    get_project_root, initialize_experiment,
)
from src.data import (
    MultiGraderDataset, PretrainingDataset, 
    get_transforms, get_ssl_transforms
)
from src.models import get_model
from src.training import TrainingEngine, PretrainingEngine

root = get_project_root()
logger = logging.getLogger(__name__)

def train(cfg, run_path):
    # -------------------- SETUP DEVICES -------------------- #
    rank = GPUSetup.get_rank()
    ngpus_per_node = torch.cuda.device_count()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f"Local Rank {local_rank}: using device {device} - Starting training setup procedure")

    if GPUSetup.is_distributed() and rank % ngpus_per_node == 0:
        print("Before DDP initialization:", flush=True)
        os.system("nvidia-smi")

    # -------------------- LOAD DATA -------------------- #
    df = pd.read_csv(cfg['paths']['data_csv'])
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    hierarchical = cfg['training']['num_subclasses'] > 0
    
    # -------------------- PRETRAINING -------------------- #
    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    if cfg['training']['ssl_pretrain']:
        logger.info(f"Starting pretraining with method: {cfg['training']['pretrain_method']}")
        pretrain_method = cfg['training']['pretrain_method']  # e.g., 'contrastive', 'mae'

        # Create base dataset without transforms
        base_dataset = MultiGraderDataset(
            video_paths=train_df[cfg['training']['datamodule']['video_col']].tolist(),
            labels=[["0"]] * len(train_df),  # Dummy labels as lists for consistency
            transform=None,
            is_ssl=True,
            hierarchical=hierarchical
        )

        # Create pretraining dataset
        ssl_transform = get_ssl_transforms(pretrain_method)
        pretrain_dataset = PretrainingDataset(
            base_dataset, 
            pretrain_method,
            cfg, 
            ssl_transform=ssl_transform
        )
        pretrain_sampler = DistributedSampler(pretrain_dataset) if GPUSetup.is_distributed() else None
        pretrain_loader = DataLoader(
            pretrain_dataset,
            batch_size=cfg['training']['batch_size'],
            shuffle=(pretrain_sampler is None),
            sampler=pretrain_sampler,
            num_workers=cfg['training']['num_workers']
        )

        # Initialize model for pretraining
        base_args = dict(
            model_name     = cfg['training']['model_name'],
            num_base_classes = cfg['training']['num_base_classes'],
            num_subclasses = 0,
            pretrained     = True,
            pretrain_method= pretrain_method,
        )
        # for MAE only
        if pretrain_method == 'mae':
            base_args.update(
                patch_size = cfg['training']['patch_size'],
                mask_ratio = cfg['training']['mask_ratio'],
                end_mask_ratio = cfg['training']['end_mask_ratio'],
            )

        model = get_model(**base_args)
        model = model.to(device)

        if GPUSetup.is_distributed():
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=True,
                find_unused_parameters=True
            )
        torch.backends.cudnn.benchmark = True

        # Set up pretraining engine
        pretrain_engine = PretrainingEngine(
            model, pretrain_loader, cfg, run_id, run_path, device, 
            pretrain_method=pretrain_method
        )
        pretrain_engine.train(cfg['training']['ssl_epochs'])

        # Save pretraining weights
        if GPUSetup.is_main_process():
            checkpoint_path = os.path.join(run_path, cfg['paths']['checkpoint_dir'], 'ssl_pretrained.pth')
            torch.save(model.module.state_dict() if GPUSetup.is_distributed() else model.state_dict(), checkpoint_path)

        # free up any cached GPU memory before we start fine-tuning
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        logger.info("Pretraining completed (cache cleared)")

    # -------------------- SUPERVISED FINE-TUNING -------------------- #
    logger.info("Starting supervised fine-tuning")
    train_transforms = get_transforms(is_train=True)
    val_transforms = get_transforms(is_train=False)
    train_dataset = MultiGraderDataset(
        video_paths=train_df[cfg['training']['datamodule']['video_col']].tolist(),
        labels=train_df[cfg['training']['datamodule']['label_cols']].values.tolist(),
        transform=train_transforms,
        hierarchical=hierarchical
    )
    val_dataset = MultiGraderDataset(
        video_paths=val_df[cfg['training']['datamodule']['video_col']].tolist(),
        labels=val_df[cfg['training']['datamodule']['label_cols']].values.tolist(),
        transform=val_transforms,
        hierarchical=hierarchical
    )

    train_sampler = DistributedSampler(train_dataset) if GPUSetup.is_distributed() else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg['training']['num_workers']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        num_workers=cfg['training']['num_workers']
    )

    # Initialize model for fine-tuning
    model = get_model(
        cfg['training']['model_name'],
        num_base_classes=cfg['training']['num_base_classes'],
        num_subclasses=cfg['training']['num_subclasses'],
        pretrained=True,
        pretrain_method=None
    )
    model = model.to(device)

    # Load pretrained weights if available
    if cfg['training']['ssl_pretrain']:
        checkpoint_path = os.path.join(run_path, cfg['paths']['checkpoint_dir'], 'ssl_pretrained.pth')
        state = torch.load(checkpoint_path, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("Missing head keys (to be randomly init'ed):", missing)
        print("Unexpected keys (ignored):", unexpected)  # Optional: log unexpected keys too
        
    if GPUSetup.is_distributed():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=True,
            find_unused_parameters=True
        )
        torch.backends.cudnn.benchmark = True

    # Define criterion, optimizer, scheduler
    if hierarchical:
        base_criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        subclass_criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        criterion = lambda outputs, batch: (
            base_criterion(outputs[0], batch['base_label'].to(device)) +
            cfg['training']['subclass_loss_weight'] * subclass_criterion(outputs[1], batch['subclass_label'].to(device))
        )
    else:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['lr'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['training']['epochs'])

    engine = TrainingEngine(
        model, optimizer, scheduler, criterion, train_loader, val_loader, cfg, run_id, run_path, device
    )
    engine.train(cfg['training']['epochs'])

    if GPUSetup.is_main_process():
        logger.info("Training completed. Final validation metrics are logged per epoch.")

    GPUSetup.cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train video classification model with multi-grader labels")
    parser.add_argument('--config_file', type=str, default='baseline_i3d_multigrader.yaml', help='Name of the config YAML file')
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', args.config_file)
    cfg = load_config(config_path)
    GPUSetup.setup(distributed=cfg.get('distributed', False),
                   seed=cfg.get('SEED', 42))

    run_path = initialize_experiment(cfg)

    # ─── configure logging (once!) ──────────────────────────────────
    log_path = os.path.join(run_path, cfg['paths']['log_dir'], 'run.log')
    logger = setup_logging(
        config_level=cfg["output_configuration"]["logging_level"],
        log_file=log_path
    )

    # ─── hand off to train() ────────────────────────────────────────
    try:
        train(cfg, run_path)
    except Exception as e:
        logger.error("Fatal error", exc_info=True)
        raise
    finally:
        GPUSetup.cleanup()



