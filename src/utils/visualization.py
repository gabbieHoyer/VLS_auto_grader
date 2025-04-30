import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch

def plot_confusion_matrix(metrics, model_save_path, run_id, title_var=None):
    os.makedirs(os.path.join(model_save_path, 'figures'), exist_ok=True)
    cm = confusion_matrix(metrics['true_labels'], metrics['pred_labels'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    if title_var:
        plt.title(f"Confusion Matrix ({title_var})")
        plt.savefig(
            os.path.join(model_save_path, 'figures', f'cm_{title_var}_{run_id}.png')
        )
    else:
        plt.title(f'Confusion Matrix')
        plt.savefig(
            os.path.join(model_save_path, 'figures', f'cm_{run_id}.png')
        )
    plt.close()


# def plot_confusion_matrix(metrics, model_save_path, run_id):
#     os.makedirs(os.path.join(model_save_path, 'figures'), exist_ok=True)
#     cm = confusion_matrix(metrics['true_labels'], metrics['pred_labels'])
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     plt.title(f'Confusion Matrix (Run {run_id})')
#     plt.savefig(os.path.join(model_save_path, 'figures', f'cm_{run_id}.png'))
#     plt.close()

def plot_loss_curves(losses, model_save_path, run_id):
    os.makedirs(os.path.join(model_save_path, 'figures'), exist_ok=True)
    plt.figure(figsize=(10, 5))
    for mode in losses:
        epochs = [x['epoch'] for x in losses[mode]]
        loss_values = [x['loss'] for x in losses[mode]]
        plt.plot(epochs, loss_values, label=f'{mode} loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(model_save_path, 'figures', f'loss_{run_id}.png'))
    plt.close()

def save_attention_maps(attention_maps, video_paths, save_dir, run_id, batch_idx):
    os.makedirs(save_dir, exist_ok=True)
    for i, (attn_map, video_path) in enumerate(zip(attention_maps, video_paths)):
        # Example: Save attention map as heatmap
        attn_map = attn_map.cpu().numpy()  # Shape: [num_heads, seq_len, seq_len] or similar
        plt.figure(figsize=(8, 6))
        sns.heatmap(attn_map.mean(axis=0), cmap='viridis')  # Average across heads
        plt.title(f'Attention Map (Video {os.path.basename(video_path)})')
        plt.savefig(os.path.join(save_dir, f'attn_map_{run_id}_batch{batch_idx}_sample{i}.png'))
        plt.close()
        # Optionally save raw attention maps as tensors
        torch.save(attn_map, os.path.join(save_dir, f'attn_map_{run_id}_batch{batch_idx}_sample{i}.pt'))