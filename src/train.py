
import os
import pandas as pd
import logging
import argparse
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader

from dataset import VideoDataset
from model import get_model, ContrastiveLoss
from metrics import compute_metrics
from utils.logger import setup_logger
from utils.augmentations import get_transforms, get_ssl_transforms
from utils.ssl_utils import ssl_pretrain
from utils.config_loader import load_config

from utils.experiment_setup import get_project_root, determine_run_directory
# from utils import GPUSetup
# from . import gpu_setup as GPUSetup

from utils import gpu_setup as GPUSetup

# from utils.logger import wandb_log

root = get_project_root()

def train(cfg):
    # --------------- SET UP ENVIRONMENT --------------- #  
    rank = GPUSetup.get_rank()
    ngpus_per_node = torch.cuda.device_count()

    # Detect if we have a GPU available and choose device accordingly
    if torch.cuda.is_available():
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')

    logger.info(f"Local Rank {local_rank}: Starting finetune_main")

    if GPUSetup.is_distributed():
        if rank % ngpus_per_node == 0:
            print("Before DDP initialization:", flush=True)
            os.system("nvidia-smi")

    # ------------- SET UP EXPERIMENT RUN  ------------- #
    module_cfg = cfg.get('training').get('module')
    datamodule_cfg = cfg.get('training').get('datamodule')
    output_cfg = cfg.get('output_configuration')

    if GPUSetup.is_distributed():
        # If distributed training is enabled, synchronize creation of the run directory
        group_name = f"{cfg.get('experiment').get('name')}"
        
        if GPUSetup.is_main_process(): 
            # Only main process determines the run directory
            run_path = determine_run_directory(output_cfg['work_dir'], output_cfg['task_name'], group_name)
            # Since run_path is a string, use broadcast_object_list
            dist.broadcast_object_list([run_path], src=0)  # src=0 denotes the main process
        else:
            # Receive broadcasted run_path
            run_path = [None]  # Placeholder for the received object
            dist.broadcast_object_list(run_path, src=0)
            run_path = run_path[0]  # Unpack the list to get the actual path
    else:
        # If not distributed, directly determine the run directory
        run_path = determine_run_directory(output_cfg['work_dir'], output_cfg['task_name'], group_name)
    
    # Initialize WandB
    if output_cfg['use_wandb']:
        if GPUSetup.is_main_process(): 
            wandb.init(project=output_cfg['task_name'], config={
                "epochs": cfg["training"]["epochs"],
                "batch_size": cfg["training"]["batch_size"],
                "learning_rate": cfg["training"]["lr"],
            })

    # Setup logging
    logger = setup_logger('train', 
            os.path.join(root, run_path, cfg['paths']['log_dir'])
            )
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    df = pd.read_csv(cfg['paths']['data_csv'])
    train_df = df[df['split'] == 'train']
    
    # SSL Pre-training
    if cfg['training']['ssl_pretrain']:
        logger.info("Starting SSL pre-training")
        ssl_transforms = get_ssl_transforms()

        # ssl_dataset = VideoDataset(
        #     video_paths=train_df['video_path'].tolist(),
        #     labels=[0] * len(train_df),  # Dummy labels (not used)
        #     transform=ssl_transforms,
        #     is_ssl=True
        # )

        ssl_dataset = VideoDataset(
            video_paths=train_df[datamodule_cfg['video_col']].tolist(),
            labels=[0] * len(train_df),  # Dummy labels (not used)
            transform=ssl_transforms,
            is_ssl=True
        )

        ssl_loader = DataLoader(ssl_dataset, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=cfg['training']['num_workers'])
        
        model = get_model(cfg['training']['model_name'], num_classes=cfg['training']['num_classes'], pretrained=True, ssl_mode=True)
        model = model.to(device)
        
        ssl_criterion = ContrastiveLoss(temperature=0.5)
        ssl_optimizer = optim.AdamW(model.parameters(), lr=cfg['training']['ssl_lr'], weight_decay=0.01)
        
        ssl_pretrain(model, ssl_loader, ssl_criterion, ssl_optimizer, cfg['training']['ssl_epochs'], device, logger)
        
        # Save pre-trained model
        # torch.save(model.state_dict(), os.path.join(cfg['paths']['checkpoint_dir'], 'ssl_pretrained.pth'))
        torch.save(model.state_dict(), 
            os.path.join(
                root, 
                run_path,
                cfg['paths']['checkpoint_dir'],
                'ssl_pretrained.pth'
            )
        )
        logger.info("SSL pre-training completed and model saved")

    # Supervised Fine-tuning
    logger.info("Starting supervised fine-tuning")
    train_transforms = get_transforms(is_train=True)

    # train_dataset = VideoDataset(
    #     video_paths=train_df['video_path'].tolist(),
    #     labels=train_df['label'].tolist(),
    #     transform=train_transforms
    # )
    train_dataset = VideoDataset(
        video_paths=train_df[datamodule_cfg['video_col']].tolist(),
        labels=train_df[datamodule_cfg['label_col']].tolist(),
        transform=train_transforms
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=cfg['training']['num_workers'])

    # Initialize model (load SSL pre-trained weights if applicable)
    model = get_model(cfg['training']['model_name'], num_classes=cfg['training']['num_classes'], pretrained=True, ssl_mode=False)
    if cfg['training']['ssl_pretrain']:
        model.load_state_dict(torch.load(os.path.join(cfg['paths']['checkpoint_dir'], 'ssl_pretrained.pth')))
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for grader disagreements
    optimizer = optim.AdamW(model.parameters(), lr=cfg['training']['lr'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['training']['epochs'])

    # Training loop
    for epoch in range(cfg['training']['epochs']):
        model.train()
        running_loss = 0.0
        for batch_idx, (videos, labels) in enumerate(train_loader):
            videos, labels = videos.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{cfg['training']['epochs']}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        scheduler.step()

        # # WandB Logging
        # wandb.log({
        #     'epoch': epoch + 1,
        #     'train_loss': avg_loss,
        #     'validation_accuracy': val_accuracy,
        #     'validation_f1': val_f1,
        #     'learning_rate': current_lr,
        #     'confusion_matrix': wandb.plot.confusion_matrix(
        #         preds=all_preds,
        #         y_true=all_labels,
        #         class_names=['Non-Procedural', 'Procedural']
        #     )
        # })
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss / len(train_loader),
        }
        # torch.save(checkpoint, os.path.join(cfg['paths']['checkpoint_dir'], f'checkpoint_epoch_{epoch+1}.pth'))
        torch.save(checkpoint, 
            os.path.join(
                root, 
                run_path,
                cfg['paths']['checkpoint_dir'],
                f'checkpoint_epoch_{epoch+1}.pth'
            )
        )
        logger.info(f"Saved checkpoint for epoch {epoch+1}")

    logger.info("Training completed")

    if output_cfg['use_wandb'] and GPUSetup.is_main_process(): 
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train video classification model with SSL pre-training")
    parser.add_argument('--config_file', type=str, default='config.yaml', help='Name of the config YAML file in ../config/')
    parser.add_argument('--data_csv', type=str, help='Path to CSV with video data')
    parser.add_argument('--model_name', type=str, choices=['i3d', 'vivit'], help='Model architecture')
    parser.add_argument('--num_classes', type=int, help='Number of classes')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--ssl_pretrain', action='store_true', help='Enable SSL pre-training')
    parser.add_argument('--ssl_epochs', type=int, help='Number of SSL pre-training epochs')
    parser.add_argument('--ssl_lr', type=float, help='SSL pre-training learning rate')
    parser.add_argument('--num_workers', type=int, help='Number of data loader workers')
    parser.add_argument('--log_dir', type=str, help='Directory for logs')
    parser.add_argument('--checkpoint_dir', type=str, help='Directory for checkpoints')
    
    args = parser.parse_args()
    
    # Construct config path
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', args.config_file)
    
    try:
        # Load config
        cfg = load_config(config_path)
        
        # Merge argparse arguments with config (argparse takes precedence)
        cfg['paths']['data_csv'] = args.data_csv or cfg['paths']['data_csv']
        cfg['paths']['log_dir'] = args.log_dir or cfg['paths']['log_dir']
        cfg['paths']['checkpoint_dir'] = args.checkpoint_dir or cfg['paths']['checkpoint_dir']
        cfg['training']['model_name'] = args.model_name or cfg['training']['model_name']
        cfg['training']['num_classes'] = args.num_classes or cfg['training']['num_classes']
        cfg['training']['batch_size'] = args.batch_size or cfg['training']['batch_size']
        cfg['training']['epochs'] = args.epochs or cfg['training']['epochs']
        cfg['training']['lr'] = args.lr or cfg['training']['lr']
        cfg['training']['ssl_pretrain'] = args.ssl_pretrain or cfg['training']['ssl_pretrain']
        cfg['training']['ssl_epochs'] = args.ssl_epochs or cfg['training']['ssl_epochs']
        cfg['training']['ssl_lr'] = args.ssl_lr or cfg['training']['ssl_lr']
        cfg['training']['num_workers'] = args.num_workers or cfg['training']['num_workers']
        
        # Create directories
        os.makedirs(cfg['paths']['log_dir'], exist_ok=True)
        os.makedirs(cfg['paths']['checkpoint_dir'], exist_ok=True)
        
        # Note: Add GPUSetup for distributed training here if needed
        GPUSetup.setup(distributed=cfg.get('distributed', True), seed=cfg.get('seed', 42))
        
        train(cfg)
    
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"An error occurred: {e}")
        raise

    finally:
        if 'GPUSetup' in locals():
            GPUSetup.cleanup()
            # log_info("Cleanup completed.")


# Run SSL Pre-training and Fine-tuning:
# Run with SSL pre-training:
# python src/train.py --data_csv data/video_data.csv --model_name i3d --num_classes 3 --batch_size 4 --epochs 50 --lr 1e-4 --ssl_pretrain --ssl_epochs 20 --ssl_lr 1e-3

# This will pre-train the model for 20 epochs using SSL, save the pre-trained weights, and then fine-tune for 50 epochs.
# To skip SSL pre-training, omit the --ssl_pretrain flag.


# -----------------------------------------------

# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from src.dataset import VideoDataset
# from src.model import get_model, ContrastiveLoss
# from src.metrics import compute_metrics
# from src.utils.logger import setup_logger
# from src.utils.augmentations import get_transforms, get_ssl_transforms
# from src.utils.ssl_utils import ssl_pretrain
# import pandas as pd
# import argparse

# def train(args):
#     # Setup logging
#     logger = setup_logger('train', args.log_dir)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logger.info(f"Using device: {device}")

#     # Load data
#     df = pd.read_csv(args.data_csv)
#     train_df = df[df['split'] == 'train']
    
#     # SSL Pre-training
#     if args.ssl_pretrain:
#         logger.info("Starting SSL pre-training")
#         ssl_transforms = get_ssl_transforms()
#         ssl_dataset = VideoDataset(
#             video_paths=train_df['video_path'].tolist(),
#             labels=[0] * len(train_df),  # Dummy labels (not used)
#             transform=ssl_transforms,
#             is_ssl=True
#         )
#         ssl_loader = DataLoader(ssl_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        
#         model = get_model(args.model_name, num_classes=args.num_classes, pretrained=True, ssl_mode=True)
#         model = model.to(device)
        
#         ssl_criterion = ContrastiveLoss(temperature=0.5)
#         ssl_optimizer = optim.AdamW(model.parameters(), lr=args.ssl_lr, weight_decay=0.01)
        
#         ssl_pretrain(model, ssl_loader, ssl_criterion, ssl_optimizer, args.ssl_epochs, device, logger)
        
#         # Save pre-trained model
#         torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'ssl_pretrained.pth'))
#         logger.info("SSL pre-training completed and model saved")

#     # Supervised Fine-tuning
#     logger.info("Starting supervised fine-tuning")
#     train_transforms = get_transforms(is_train=True)
#     train_dataset = VideoDataset(
#         video_paths=train_df['video_path'].tolist(),
#         labels=train_df['label'].tolist(),
#         transform=train_transforms
#     )
#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

#     # Initialize model (load SSL pre-trained weights if applicable)
#     model = get_model(args.model_name, num_classes=args.num_classes, pretrained=True, ssl_mode=False)
#     if args.ssl_pretrain:
#         model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'ssl_pretrained.pth')))
#     model = model.to(device)
    
#     # Loss and optimizer
#     criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for grader disagreements
#     optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

#     # Training loop
#     for epoch in range(args.epochs):
#         model.train()
#         running_loss = 0.0
#         for batch_idx, (videos, labels) in enumerate(train_loader):
#             videos, labels = videos.to(device), labels.to(device)
            
#             optimizer.zero_grad()
#             outputs = model(videos)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
            
#             running_loss += loss.item()
            
#             if batch_idx % 10 == 0:
#                 logger.info(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
#         scheduler.step()
        
#         # Save checkpoint
#         checkpoint = {
#             'epoch': epoch + 1,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': running_loss / len(train_loader),
#         }
#         torch.save(checkpoint, os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
#         logger.info(f"Saved checkpoint for epoch {epoch+1}")

#     logger.info("Training completed")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Train video classification model with SSL pre-training")
#     parser.add_argument('--data_csv', type=str, default='data/video_data.csv', help='Path to CSV with video data')
#     parser.add_argument('--model_name', type=str, choices=['i3d', 'vivit'], default='i3d', help='Model architecture')
#     parser.add_argument('--num_classes', type=int, default=3, help='Number of classes')
#     parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
#     parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
#     parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
#     parser.add_argument('--ssl_pretrain', action='store_true', help='Enable SSL pre-training')
#     parser.add_argument('--ssl_epochs', type=int, default=20, help='Number of SSL pre-training epochs')
#     parser.add_argument('--ssl_lr', type=float, default=1e-3, help='SSL pre-training learning rate')
#     parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
#     parser.add_argument('--log_dir', type=str, default='logs/', help='Directory for logs')
#     parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/', help='Directory for checkpoints')
    
#     args = parser.parse_args()
    
#     os.makedirs(args.log_dir, exist_ok=True)
#     os.makedirs(args.checkpoint_dir, exist_ok=True)
    
#     train(args)

# Run SSL Pre-training and Fine-tuning:
# Run with SSL pre-training:
# python src/train.py --data_csv data/video_data.csv --model_name i3d --num_classes 3 --batch_size 4 --epochs 50 --lr 1e-4 --ssl_pretrain --ssl_epochs 20 --ssl_lr 1e-3

# This will pre-train the model for 20 epochs using SSL, save the pre-trained weights, and then fine-tune for 50 epochs.
# To skip SSL pre-training, omit the --ssl_pretrain flag.

# ----------------------------------------------
#  original without pretraining

# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from src.dataset import VideoDataset
# from src.model import get_model
# from src.metrics import compute_metrics
# from src.utils.logger import setup_logger
# from src.utils.augmentations import get_transforms
# import pandas as pd
# import argparse

# def train(args):
#     # Setup logging
#     logger = setup_logger('train', args.log_dir)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     logger.info(f"Using device: {device}")

#     # Load data
#     df = pd.read_csv(args.data_csv)
#     train_df = df[df['split'] == 'train']
    
#     train_transforms = get_transforms(is_train=True)
#     train_dataset = VideoDataset(
#         video_paths=train_df['video_path'].tolist(),
#         labels=train_df['label'].tolist(),
#         transform=train_transforms
#     )
#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

#     # Initialize model
#     model = get_model(args.model_name, num_classes=args.num_classes, pretrained=True)
#     model = model.to(device)
    
#     # Loss and optimizer
#     criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for grader disagreements
#     optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

#     # Training loop
#     for epoch in range(args.epochs):
#         model.train()
#         running_loss = 0.0
#         for batch_idx, (videos, labels) in enumerate(train_loader):
#             videos, labels = videos.to(device), labels.to(device)
            
#             optimizer.zero_grad()
#             outputs = model(videos)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
            
#             running_loss += loss.item()
            
#             if batch_idx % 10 == 0:
#                 logger.info(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
#         scheduler.step()
        
#         # Save checkpoint
#         checkpoint = {
#             'epoch': epoch + 1,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': running_loss / len(train_loader),
#         }
#         torch.save(checkpoint, os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
#         logger.info(f"Saved checkpoint for epoch {epoch+1}")

#     logger.info("Training completed")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Train video classification model")
#     parser.add_argument('--data_csv', type=str, default='data/video_data.csv', help='Path to CSV with video data')
#     parser.add_argument('--model_name', type=str, choices=['i3d', 'vivit'], default='i3d', help='Model architecture')
#     parser.add_argument('--num_classes', type=int, default=3, help='Number of classes')
#     parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
#     parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
#     parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
#     parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
#     parser.add_argument('--log_dir', type=str, default='logs/', help='Directory for logs')
#     parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/', help='Directory for checkpoints')
    
#     args = parser.parse_args()
    
#     os.makedirs(args.log_dir, exist_ok=True)
#     os.makedirs(args.checkpoint_dir, exist_ok=True)
    
#     train(args)


# python src/train.py --data_csv data/video_data.csv --model_name i3d --num_classes 3 --batch_size 4 --epochs 50 --lr 1e-4