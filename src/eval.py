import os
import argparse
import logging
import torch
from torch.utils.data import DataLoader, DistributedSampler
import pandas as pd

from utils import GPUSetup, setup_logger, load_config, determine_run_directory, get_project_root
from dataset import VideoDataset
from model import get_model
from engine import EvaluationEngine
from utils.augmentations import get_transforms

root = get_project_root()
logger = logging.getLogger(__name__)

def evaluate(cfg):
    rank = GPUSetup.get_rank()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    # Setup experiment run
    output_cfg = cfg['output_configuration']
    if GPUSetup.is_distributed():
        torch.distributed.barrier()
        if GPUSetup.is_main_process():
            run_path = determine_run_directory(output_cfg['work_dir'], output_cfg['task_name'], cfg)
            torch.distributed.broadcast_object_list([run_path], src=0)
        else:
            run_path = [None]
            torch.distributed.broadcast_object_list(run_path, src=0)
            run_path = run_path[0]
    else:
        run_path = determine_run_directory(output_cfg['work_dir'], output_cfg['task_name'], cfg)

    # Create output subdirectories under run_path
    if GPUSetup.is_main_process():
        os.makedirs(os.path.join(run_path, cfg['paths']['log_dir']), exist_ok=True)
        os.makedirs(os.path.join(run_path, cfg['paths']['figures_dir']), exist_ok=True)

    logger = setup_logger('eval', os.path.join(run_path, cfg['paths']['log_dir']))
    logger.info(f"Using device: {device}")

    # Load data
    df = pd.read_csv(cfg['paths']['data_csv'])
    eval_df = df[df['split'] == cfg['evaluation']['split']]  # 'val' or 'test'
    eval_transforms = get_transforms(is_train=False)
    eval_dataset = VideoDataset(
        video_paths=eval_df[cfg['training']['datamodule']['video_col']].tolist(),
        labels=eval_df[cfg['training']['datamodule']['label_col']].tolist(),
        transform=eval_transforms
    )
    eval_sampler = DistributedSampler(eval_dataset, shuffle=False) if GPUSetup.is_distributed() else None
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        sampler=eval_sampler,
        num_workers=cfg['training']['num_workers']
    )

    # Initialize model
    model = get_model(cfg['training']['model_name'], num_classes=cfg['training']['num_classes'], pretrained=False)
    checkpoint_path = os.path.join(run_path, cfg['evaluation']['checkpoint_path'])
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        logger.error(f"Checkpoint not found at {checkpoint_path}")
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    if GPUSetup.is_distributed():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    model = model.to(device)

    # Initialize evaluation engine
    engine = EvaluationEngine(
        model=model,
        eval_loader=eval_loader,
        cfg=cfg,
        run_path=run_path,
        device=device,
        save_attention=cfg['evaluation']['save_attention']
    )
    metrics = engine.evaluate()

    if GPUSetup.is_main_process():
        logger.info(f"Evaluation Metrics ({cfg['evaluation']['split']} split):")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        if cfg['output_configuration'].get('use_wandb'):
            import wandb
            wandb.finish()

    GPUSetup.cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate video classification model")
    parser.add_argument('--config_file', type=str, default='config.yaml', help='Name of the config YAML file')
    parser.add_argument('--data_csv', type=str, help='Path to CSV with video data')
    parser.add_argument('--model_name', type=str, choices=['i3d', 'vivit'], help='Model architecture')
    parser.add_argument('--num_classes', type=int, help='Number of classes')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--num_workers', type=int, help='Number of data loader workers')
    parser.add_argument('--split', type=str, choices=['val', 'test'], help='Split to evaluate')
    parser.add_argument('--checkpoint_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--log_dir', type=str, help='Directory for logs')
    parser.add_argument('--save_attention', action='store_true', help='Save attention maps for ViViT')
    
    args = parser.parse_args()
    
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', args.config_file)
    cfg = load_config(config_path)
    
    # Merge argparse arguments with config
    cfg['paths']['data_csv'] = args.data_csv or cfg['paths']['data_csv']
    cfg['paths']['log_dir'] = args.log_dir or cfg['paths']['log_dir']
    cfg['training']['model_name'] = args.model_name or cfg['training']['model_name']
    cfg['training']['num_classes'] = args.num_classes or cfg['training']['num_classes']
    cfg['training']['batch_size'] = args.batch_size or cfg['training']['batch_size']
    cfg['training']['num_workers'] = args.num_workers or cfg['training']['num_workers']
    cfg['evaluation']['split'] = args.split or cfg['evaluation']['split']
    cfg['evaluation']['checkpoint_path'] = args.checkpoint_path or cfg.get('evaluation', {}).get('checkpoint_path', '')
    cfg['evaluation']['save_attention'] = args.save_attention or cfg['evaluation']['save_attention']
    
    GPUSetup.setup(distributed=cfg.get('distributed', True), seed=cfg.get('SEED', 42))
    try:
        evaluate(cfg)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise
    finally:
        GPUSetup.cleanup()
