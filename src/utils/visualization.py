import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch
import numpy as np

# -------- AUGMENTATIONS FIGURES -------- #

# These must match whatever Normalize() parameters you used in your A.Compose
MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

def _denormalize(vid):
    """
    vid: numpy array shaped [T, H, W, C] in normalized space
    returns: clipped [T, H, W, C] in [0..1]
    """
    vid = vid * STD + MEAN
    return np.clip(vid, 0, 1)


def plot_augmentations(loader,
                       run_path,
                       fig_dir,
                       run_id,
                       prefix='train',
                       n_frames=4,
                       save_path=None,
                       close_fig=True):
    """
    Supervised viz: assumes batch is a dict with 'video' or tuple (video, labels).
    
    Args:
        loader: DataLoader providing the batch.
        run_path: Base path for saving figures.
        fig_dir: Directory for figures relative to run_path.
        run_id: Unique identifier for the run.
        prefix: Prefix for the figure filename ('train' or 'val').
        n_frames: Number of frames to display.
        save_path: Optional path to save the figure. If None, figure is not saved.
        close_fig: If True, close the figure after saving (set False for inline display).
    
    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    batch = next(iter(loader))
    vids = batch['video'] if isinstance(batch, dict) else batch[0]
    # pick the first sample, permute to [T, H, W, C]
    vid = vids[0].cpu().permute(1, 2, 3, 0).numpy()
    vid = _denormalize(vid)

    T = vid.shape[0]
    idxs = np.linspace(0, T - 1, n_frames, dtype=int)

    fig, axes = plt.subplots(1, n_frames, figsize=(n_frames * 3, 3))
    for ax, idx in zip(axes, idxs):
        ax.imshow(vid[idx])
        ax.axis('off')
        ax.set_title(f'frame {idx}')
    fig.suptitle(f'{prefix.capitalize()} augment sample', y=1.05)

    if save_path:
        out_dir = os.path.join(run_path, fig_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = save_path if save_path else os.path.join(out_dir, f'{prefix}_augment_{run_id}.png')
        fig.savefig(out_path, bbox_inches='tight')

    if close_fig:
        plt.close(fig)

    return fig


def plot_pretrain_augmentations(loader,
                                run_path,
                                fig_dir,
                                run_id,
                                pretrain_method,
                                n_frames=4,
                                save_path=None,
                                close_fig=True):
    """
    SSL viz: supports contrastive, moco, mae modes.
    
    Args:
        loader: DataLoader providing the batch.
        run_path: Base path for saving figures.
        fig_dir: Directory for figures relative to run_path.
        run_id: Unique identifier for the run.
        pretrain_method: Pretraining method ('contrastive', 'moco', 'mae').
        n_frames: Number of frames to display.
        save_path: Optional path to save the figure. If None, figure is not saved.
        close_fig: If True, close the figure after saving (set False for inline display).
    
    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    batch = next(iter(loader))

    def to_vid_np(x):  # x: [C, T, H, W]
        arr = x[0].cpu().permute(1, 2, 3, 0).numpy()
        return _denormalize(arr)

    if pretrain_method in ('contrastive', 'moco'):
        v1 = to_vid_np(batch['video1'])
        v2 = to_vid_np(batch['video2']) if pretrain_method == 'contrastive' else None
    else:  # mae
        v1 = to_vid_np(batch['original_video'])
        v2 = to_vid_np(batch['masked_video'])

    T = v1.shape[0]
    idxs = np.linspace(0, T - 1, n_frames, dtype=int)
    rows = 2 if v2 is not None else 1

    fig, axes = plt.subplots(rows, n_frames,
                             figsize=(n_frames * 3, rows * 3))
    axes = np.atleast_2d(axes)

    # first row
    for j, idx in enumerate(idxs):
        axes[0, j].imshow(v1[idx])
        axes[0, j].axis('off')
        axes[0, j].set_title(f'f{idx}')

    # second row if needed
    if v2 is not None:
        for j, idx in enumerate(idxs):
            axes[1, j].imshow(v2[idx])
            axes[1, j].axis('off')
            axes[1, j].set_title(f'f{idx}')

    fig.suptitle(f"Pretrain [{pretrain_method}] augment", y=1.02)

    if save_path:
        out_dir = os.path.join(run_path, fig_dir)
        os.makedirs(out_dir, exist_ok=True)
        fname = f"pretrain_augment_{pretrain_method}_{run_id}.png"
        out_path = save_path if save_path else os.path.join(out_dir, fname)
        fig.savefig(out_path, bbox_inches='tight')

    if close_fig:
        plt.close(fig)

    return fig


# -------- METRICS FIGURES -------- #

def plot_confusion_matrix(metrics, model_save_path, run_id, title_var=None):
    """
    Plot and save a confusion matrix.
    
    Args:
        metrics: Dictionary with 'true_labels' and 'pred_labels'.
        model_save_path: Base path for saving figures.
        run_id: Unique identifier for the run.
        title_var: Optional title suffix.
    """
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
    
# def plot_augmentations(loader,
#                        run_path,
#                        fig_dir,
#                        run_id,
#                        prefix='train',
#                        n_frames=4):
#     """
#     Supervised viz: assumes batch is a dict with 'video' or tuple (video, labels).
#     """
#     batch = next(iter(loader))
#     vids = batch['video'] if isinstance(batch, dict) else batch[0]
#     # pick the first sample, permute to [T, H, W, C]
#     vid = vids[0].cpu().permute(1, 2, 3, 0).numpy()
#     vid = _denormalize(vid)

#     T = vid.shape[0]
#     idxs = np.linspace(0, T - 1, n_frames, dtype=int)

#     fig, axes = plt.subplots(1, n_frames, figsize=(n_frames * 3, 3))
#     for ax, idx in zip(axes, idxs):
#         ax.imshow(vid[idx])
#         ax.axis('off')
#         ax.set_title(f'frame {idx}')
#     fig.suptitle(f'{prefix.capitalize()} augment sample', y=1.05)

#     out_dir = os.path.join(run_path, fig_dir)
#     os.makedirs(out_dir, exist_ok=True)
#     out_path = os.path.join(out_dir, f'{prefix}_augment_{run_id}.png')
#     fig.savefig(out_path, bbox_inches='tight')
#     plt.close(fig)


# def plot_pretrain_augmentations(loader,
#                                 run_path,
#                                 fig_dir,
#                                 run_id,
#                                 pretrain_method,
#                                 n_frames=4):
#     """
#     SSL viz: supports contrastive, moco, mae modes.
#     """
#     batch = next(iter(loader))

#     def to_vid_np(x):  # x: [C, T, H, W]
#         arr = x[0].cpu().permute(1, 2, 3, 0).numpy()
#         return _denormalize(arr)

#     if pretrain_method in ('contrastive', 'moco'):
#         v1 = to_vid_np(batch['video1'])
#         v2 = to_vid_np(batch['video2']) if pretrain_method == 'contrastive' else None
#     else:  # mae
#         v1 = to_vid_np(batch['original_video'])
#         v2 = to_vid_np(batch['masked_video'])

#     T = v1.shape[0]
#     idxs = np.linspace(0, T - 1, n_frames, dtype=int)
#     rows = 2 if v2 is not None else 1

#     fig, axes = plt.subplots(rows, n_frames,
#                              figsize=(n_frames * 3, rows * 3))
#     axes = np.atleast_2d(axes)

#     # first row
#     for j, idx in enumerate(idxs):
#         axes[0, j].imshow(v1[idx])
#         axes[0, j].axis('off')
#         axes[0, j].set_title(f'f{idx}')

#     # second row if needed
#     if v2 is not None:
#         for j, idx in enumerate(idxs):
#             axes[1, j].imshow(v2[idx])
#             axes[1, j].axis('off')
#             axes[1, j].set_title(f'f{idx}')

#     fig.suptitle(f"Pretrain [{pretrain_method}] augment", y=1.02)

#     out_dir = os.path.join(run_path, fig_dir)
#     os.makedirs(out_dir, exist_ok=True)
#     fname = f"pretrain_augment_{pretrain_method}_{run_id}.png"
#     fig.savefig(os.path.join(out_dir, fname), bbox_inches='tight')
#     plt.close(fig)



# -------- METRICS FIGURES -------- #

# def plot_confusion_matrix(metrics, model_save_path, run_id, title_var=None):
#     os.makedirs(os.path.join(model_save_path, 'figures'), exist_ok=True)
#     cm = confusion_matrix(metrics['true_labels'], metrics['pred_labels'])
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     if title_var:
#         plt.title(f"Confusion Matrix ({title_var})")
#         plt.savefig(
#             os.path.join(model_save_path, 'figures', f'cm_{title_var}_{run_id}.png')
#         )
#     else:
#         plt.title(f'Confusion Matrix')
#         plt.savefig(
#             os.path.join(model_save_path, 'figures', f'cm_{run_id}.png')
#         )
#     plt.close()


# def plot_confusion_matrix(metrics, model_save_path, run_id):
#     os.makedirs(os.path.join(model_save_path, 'figures'), exist_ok=True)
#     cm = confusion_matrix(metrics['true_labels'], metrics['pred_labels'])
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     plt.title(f'Confusion Matrix (Run {run_id})')
#     plt.savefig(os.path.join(model_save_path, 'figures', f'cm_{run_id}.png'))
#     plt.close()

# -------- LOSS CURVES FIGURES -------- #

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

# -------- ATTENTION MAPS FIGURES -------- #

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