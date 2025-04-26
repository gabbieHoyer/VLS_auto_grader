import os
import pandas as pd
import logging
import argparse

import torch
from torch.utils.data import DataLoader

from src.dataset import VideoDataset
from src.model import get_model
from src.metrics import compute_metrics
from src.utils.logger import setup_logger
from src.utils.augmentations import get_transforms
from src.utils.config_loader import load_config

# from . import gpu_setup as GPUSetup
from utils.logger import wandb_log

from src.utils import get_project_root, determine_run_directory, GPUSetup

root = get_project_root()

def evaluate(config):
    # Setup logging
    logger = setup_logger('eval', config['paths']['log_dir'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    df = pd.read_csv(config['paths']['data_csv'])
    eval_df = df[df['split'] == config['evaluation']['split']]
    
    eval_transforms = get_transforms(is_train=False)
    eval_dataset = VideoDataset(
        video_paths=eval_df['video_path'].tolist(),
        labels=eval_df['label'].tolist(),
        transform=eval_transforms
    )
    eval_loader = DataLoader(eval_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['training']['num_workers'])

    # Initialize model
    model = get_model(config['training']['model_name'], num_classes=config['training']['num_classes'], pretrained=False)
    checkpoint = torch.load(config['evaluation']['checkpoint_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Evaluation
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for videos, labels in eval_loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Save attention maps for ViViT (if applicable)
            if config['training']['model_name'] == 'vivit' and config['evaluation']['save_attention']:
                attention_maps = model.get_attention_maps(videos)  # Implement in model.py
                # Save attention_maps (placeholder for your implementation)
                logger.info("Attention maps saved (placeholder)")

    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds, config['training']['num_classes'])
    logger.info(f"Evaluation Metrics ({config['evaluation']['split']} split):")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate video classification model")
    parser.add_argument('--config_file', type=str, default='config.yaml', help='Name of the config YAML file in ../config/')
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
    
    # Construct config path
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', args.config_file)
    
    try:
        # Load config
        config = load_config(config_path)
        
        # Merge argparse arguments with config (argparse takes precedence)
        config['paths']['data_csv'] = args.data_csv or config['paths']['data_csv']
        config['paths']['log_dir'] = args.log_dir or config['paths']['log_dir']
        config['training']['model_name'] = args.model_name or config['training']['model_name']
        config['training']['num_classes'] = args.num_classes or config['training']['num_classes']
        config['training']['batch_size'] = args.batch_size or config['training']['batch_size']
        config['training']['num_workers'] = args.num_workers or config['training']['num_workers']
        config['evaluation']['split'] = args.split or config['evaluation']['split']
        config['evaluation']['checkpoint_path'] = args.checkpoint_path or config.get('evaluation', {}).get('checkpoint_path', '')
        config['evaluation']['save_attention'] = args.save_attention or config['evaluation']['save_attention']
        
        # Create directories
        os.makedirs(config['paths']['log_dir'], exist_ok=True)
        
        # Note: Add GPUSetup for distributed training here if needed
        # e.g., GPUSetup.setup(distributed=True, seed=42)
        
        evaluate(config)
    
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"An error occurred: {e}")
        raise

# ------------------------------------------------------

# import os
# import torch
# from torch.utils.data import DataLoader
# from src.dataset import VideoDataset
# from src.model import get_model
# from src.metrics import compute_metrics
# from src.utils.logger import setup_logger
# from src.utils.augmentations import get_transforms
# import pandas as pd
# import argparse

# def evaluate(args):
#     # Setup logging
#     logger = setup_logger('eval', args.log_dir)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logger.info(f"Using device: {device}")

#     # Load data
#     df = pd.read_csv(args.data_csv)
#     eval_df = df[df['split'] == args.split]
    
#     eval_transforms = get_transforms(is_train=False)
#     eval_dataset = VideoDataset(
#         video_paths=eval_df['video_path'].tolist(),
#         labels=eval_df['label'].tolist(),
#         transform=eval_transforms
#     )
#     eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

#     # Initialize model
#     model = get_model(args.model_name, num_classes=args.num_classes, pretrained=False)
#     checkpoint = torch.load(args.checkpoint_path, map_location=device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model = model.to(device)
#     model.eval()

#     # Evaluation
#     all_preds = []
#     all_labels = []
#     with torch.no_grad():
#         for videos, labels in eval_loader:
#             videos, labels = videos.to(device), labels.to(device)
#             outputs = model(videos)
#             _, preds = torch.max(outputs, 1)
            
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
            
#             # Save attention maps for ViViT (if applicable)
#             if args.model_name == 'vivit' and args.save_attention:
#                 attention_maps = model.get_attention_maps(videos)  # Implement in model.py
#                 # Save attention_maps (placeholder for your implementation)
#                 logger.info("Attention maps saved (placeholder)")

#     # Compute metrics
#     metrics = compute_metrics(all_labels, all_preds, args.num_classes)
#     logger.info(f"Evaluation Metrics ({args.split} split):")
#     for metric, value in metrics.items():
#         logger.info(f"{metric}: {value:.4f}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Evaluate video classification model")
#     parser.add_argument('--data_csv', type=str, default='data/video_data.csv', help='Path to CSV with video data')
#     parser.add_argument('--model_name', type=str, choices=['i3d', 'vivit'], default='i3d', help='Model architecture')
#     parser.add_argument('--num_classes', type=int, default=3, help='Number of classes')
#     parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
#     parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
#     parser.add_argument('--split', type=str, choices=['val', 'test'], default='val', help='Split to evaluate')
#     parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
#     parser.add_argument('--log_dir', type=str, default='logs/', help='Directory for logs')
#     parser.add_argument('--save_attention', action='store_true', help='Save attention maps for ViViT')
    
#     args = parser.parse_args()
    
#     os.makedirs(args.log_dir, exist_ok=True)
    
#     evaluate(args)

# # python src/eval.py --data_csv data/video_data.csv --model_name i3d --num_classes 3 --split test --checkpoint_path checkpoints/checkpoint_epoch_50.

# # Use the existing eval.py script to evaluate the fine-tuned model:
# # python src/eval.py --data_csv data/video_data.csv --model_name i3d --num_classes 3 --split test --checkpoint_path checkpoints/checkpoint_epoch_50.pth