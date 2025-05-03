# src/utils/checkpointing.py
import torch
import os

def log_and_checkpoint(model, optimizer, scheduler, epoch, metric, model_save_path, run_id, is_best=False):
    """
    Saves model checkpoint and logs the operation.
    """
    # Local import to avoid circular dependency
    from .gpu_setup import GPUSetup

    if not GPUSetup.is_main_process():
        return

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metric': metric
    }

    checkpoint_dir = os.path.join(model_save_path, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save the checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    GPUSetup.save_on_master(checkpoint, checkpoint_path)

    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        GPUSetup.save_on_master(checkpoint, best_path)
        print(f"Saved best model checkpoint at {best_path}")

def final_checkpoint_conversion(model_save_path, run_id):
    """
    Converts the best checkpoint to a format suitable for inference.
    """
    # Local import to avoid circular dependency
    from .gpu_setup import GPUSetup

    if not GPUSetup.is_main_process():
        return

    best_checkpoint_path = os.path.join(model_save_path, 'checkpoints', 'best_model.pth')
    if not os.path.exists(best_checkpoint_path):
        print(f"No best checkpoint found at {best_checkpoint_path}")
        return

    checkpoint = torch.load(best_checkpoint_path, map_location='cpu')
    model_state_dict = checkpoint['model_state_dict']

    converted_path = os.path.join(model_save_path, 'checkpoints', 'converted_best_model.pth')
    GPUSetup.save_on_master(model_state_dict, converted_path)
    print(f"Converted best model saved at {converted_path}")


# import torch

# from . import GPUSetup

# def log_and_checkpoint(mode, train_loss, val_loss, module_cfg, model, optimizer, epoch,
#                        model_save_path, run_id, best_val_metric, metrics):
#     if not GPUSetup.is_main_process():
#         return best_val_metric

#     checkpoint = {
#         'epoch': epoch + 1,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'train_loss': train_loss,
#         'val_loss': val_loss,
#         'metrics': metrics
#     }
#     torch.save(checkpoint, os.path.join(model_save_path, f'checkpoint_epoch_{epoch+1}.pth'))
#     if metrics.get('accuracy', 0.0) > best_val_metric:
#         torch.save(checkpoint, os.path.join(model_save_path, f'best_model_{run_id}.pth'))
#         best_val_metric = metrics.get('accuracy', 0.0)
#     return best_val_metric

# def final_checkpoint_conversion(cfg, model_save_path, run_id):
#     if cfg['output_configuration'].get('always_convert_best') and GPUSetup.is_main_process():
#         # Convert best model to a specific format if needed
#         pass


# -------------------------------------------------------

# import os
# from collections import OrderedDict
# from datetime import datetime
# from os.path import join
# import torch
# import torch.nn as nn
# from tqdm import tqdm
# import wandb

# from . import gpu_setup as GPUSetup
# from .logger import wandb_log

# def log_and_checkpoint(mode, train_loss, val_loss, module_cfg, model, optimizer,
#                        epoch, model_save_path, run_id, best_train_loss, best_val_loss):
#     if not GPUSetup.is_main_process():
#         return best_train_loss, best_val_loss

#     # Extract user-specified checkpoint interval (default to 5)
#     checkpoint_interval = module_cfg.get("checkpoint_interval", 5)

#     # Construct checkpoint paths
#     latest_checkpoint_path = os.path.join(
#         model_save_path, f"{run_id}_finetuned_model_latest_epoch_{epoch+1}.pth"
#     )
#     best_checkpoint_path = os.path.join(
#         model_save_path, f"{run_id}_finetuned_model_best.pth"
#     )

#     # Log losses to Weights & Biases if applicable
#     if module_cfg.get('use_wandb', False):
#         if mode == 'both':
#             wandb_log({"train_epoch_loss": train_loss, "val_epoch_loss": val_loss})
#         else:
#             wandb_log({f"{mode}_epoch_loss": train_loss if mode == 'train' else val_loss})

#     # Print epoch loss
#     print(
#         f"Time: {datetime.now().strftime('%Y%m%d-%H%M')}, "
#         f"Training Loss: {train_loss}, Validation Loss: {val_loss}",
#         flush=True
#     )

#     # Save a "latest" checkpoint only if (epoch + 1) is a multiple of checkpoint_interval
#     if (epoch + 1) % checkpoint_interval == 0:
#         torch.save({
#             "model": model.state_dict(),
#             "optimizer": optimizer.state_dict(),
#             "epoch": epoch,
#             "train_loss": train_loss,
#             "val_loss": val_loss
#         }, latest_checkpoint_path)

#     # Always update and save the best checkpoint based on validation loss
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss  # Update best validation loss
#         torch.save({
#             "model": model.state_dict(),
#             "optimizer": optimizer.state_dict(),
#             "epoch": epoch,
#             "loss": val_loss,
#         }, best_checkpoint_path)

#     if train_loss < best_train_loss:
#         best_train_loss = train_loss  # Update best training loss

#     return best_train_loss, best_val_loss

# # Function to check if the 'module.' prefix is present
# def check_module_prefix(state_dict):
#     return any(k.startswith('module.') for k in state_dict.keys())

# def final_checkpoint_conversion(module_cfg, model_save_path, run_id):
#     # Check flag to decide whether to convert the final checkpoint; default True
#     convert_final = module_cfg.get("convert_final_checkpoint", True)
#     if not convert_final:
#         return

#     best_checkpoint_path = os.path.join(
#         model_save_path, f"{run_id}_finetuned_model_best.pth"
#     )
#     converted_checkpoint_path = os.path.join(
#         model_save_path, f"{run_id}_finetuned_model_best_converted.pth"
#     )

#     # Load the fine-tuned checkpoint
#     finetuned_ckpt = torch.load(best_checkpoint_path)
    
#     # Correct the 'model' keys if saved from a multi-GPU setup
#     if 'model' in finetuned_ckpt and check_module_prefix(finetuned_ckpt['model']):
#         new_model_state_dict = OrderedDict()
#         for k, v in finetuned_ckpt['model'].items():
#             new_key = k[7:] if k.startswith('module.') else k
#             new_model_state_dict[new_key] = v
#         finetuned_ckpt['model'] = new_model_state_dict
    
#     # Ensure the directory exists
#     os.makedirs(os.path.dirname(converted_checkpoint_path), exist_ok=True)
    
#     # Save the updated (converted) checkpoint
#     torch.save(finetuned_ckpt, converted_checkpoint_path)

#     # Log the converted checkpoint path to Weights & Biases if applicable
#     if module_cfg.get('use_wandb', False):
#         wandb_log({"final_checkpoint_path": converted_checkpoint_path})