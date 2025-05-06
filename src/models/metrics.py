# src/models/metrics.py
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def compute_metrics(true_labels, pred_labels, num_classes):
    """
    Compute classification metrics for true and predicted labels.

    Args:
        true_labels (torch.Tensor or list): True labels (shape: [N]).
        pred_labels (torch.Tensor or list): Predicted labels (shape: [N]).
        num_classes (int): Number of classes.

    Returns:
        dict: Dictionary of metrics (accuracy, f1, precision, recall, and per-class F1).
    """
    # Convert to NumPy if inputs are tensors
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()
    if isinstance(pred_labels, torch.Tensor):
        pred_labels = pred_labels.cpu().numpy()

    # Ensure inputs are 1D arrays
    true_labels = np.asarray(true_labels).flatten()
    pred_labels = np.asarray(pred_labels).flatten()

    # Compute weighted metrics
    metrics = {
        'accuracy': accuracy_score(true_labels, pred_labels),
        'f1_score': f1_score(true_labels, pred_labels, average='weighted', zero_division=0),
        'precision': precision_score(true_labels, pred_labels, average='weighted', zero_division=0),
        'recall': recall_score(true_labels, pred_labels, average='weighted', zero_division=0),
    }

    # Compute per-class F1 scores (optional, for detailed analysis)
    f1_per_class = f1_score(true_labels, pred_labels, average=None, zero_division=0)
    metrics['f1_per_class'] = f1_per_class.tolist()  # Convert to list for logging

    return metrics


# Mapping for reconstructing labels (for metrics in hierarchical mode)
BASE_CLASSES = {0: '1', 1: '2', 2: '3', 3: '4'}
SUBCLASSES = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: ''}

def reconstruct_label(base_pred, subclass_pred):
    base = BASE_CLASSES[base_pred]
    subclass = SUBCLASSES[subclass_pred]
    return f"{base}{subclass}" if subclass else base

def compute_multi_grader_metrics(preds, labels, num_classes, hierarchical=False, base_preds=None, subclass_preds=None):
    if hierarchical:
        # Reconstruct predicted labels
        pred_labels = [reconstruct_label(base, subclass) for base, subclass in zip(base_preds, subclass_preds)]
    else:
        pred_labels = preds

    # Parse ground truth labels for each grader
    sean_labels = []
    santiago_labels = []
    if hierarchical:
        for label_pair in labels:
            sean_base, sean_sub = parse_label(label_pair[0])
            santiago_base, santiago_sub = parse_label(label_pair[1])
            sean_labels.append(f"{sean_base}{SUBCLASSES[sean_sub]}" if sean_sub != 4 else str(sean_base))
            santiago_labels.append(f"{santiago_base}{SUBCLASSES[santiago_sub]}" if santiago_sub != 4 else str(santiago_base))
    else:
        # Labels are already mapped integers
        sean_labels = [label_pair[0] for label_pair in labels]
        santiago_labels = [label_pair[1] for label_pair in labels]

    # Map to SEAN_CLASS_NAMES for consistency
    SEAN_CLASS_NAMES = ['1', '2', '2b', '2c', '3', '3b', '3c', '4b']
    SEAN_CLASS_TO_IDX = {name: idx for idx, name in enumerate(SEAN_CLASS_NAMES)}

    if hierarchical:
        sean_targets = []
        santiago_targets = []
        preds_mapped = []
        for pred, sean, santiago in zip(pred_labels, sean_labels, santiago_labels):
            if pred == '2d':  # Map 2d to 2 for consistency with Sean_Review
                pred = '2'
            if sean == '2d':
                sean = '2'
            if santiago == '2d':
                santiago = '2'
            preds_mapped.append(SEAN_CLASS_TO_IDX.get(pred, 0))
            sean_targets.append(SEAN_CLASS_TO_IDX.get(sean, 0))
            santiago_targets.append(SEAN_CLASS_TO_IDX.get(santiago, 0))
    else:
        sean_targets = sean_labels
        santiago_targets = santiago_labels
        preds_mapped = preds

    sean_metrics = compute_metrics(sean_targets, preds_mapped, num_classes)
    santiago_metrics = compute_metrics(santiago_targets, preds_mapped, num_classes)
    return sean_metrics, santiago_metrics

def parse_label(label):
    # Simplified parse_label for metrics (matches dataset_multigrader.py)
    base_str = ''.join(c for c in label if c.isdigit())
    subclass_str = ''.join(c for c in label if c.isalpha()) or 'none'
    base = int(base_str)
    subclasses = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'none': 4}
    subclass = subclasses[subclass_str]
    return base, subclass


# --------------------------------------------

def flatten_metrics(prefix: str, metrics: dict, out: dict):
    """
    Given a metrics dict like {'sean_accuracy':0.8, 'sean_f1':tensor([0.7,0.6,0.9])},
    write into `out` keys like 'train_sean_accuracy' and
    'train_sean_f1_class_0', 'train_sean_f1_class_1', …
    """
    for name, value in metrics.items():
        # convert torch.Tensor to numpy or float
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        # array-like: unroll per class
        if hasattr(value, '__len__') and not isinstance(value, (str, bytes)):
            for i, v in enumerate(value):
                out[f'{prefix}_{name}_class_{i}'] = float(v)
        else:
            # single number
            out[f'{prefix}_{name}'] = float(value)


# def flatten_with_names(self, prefix, metrics, out_dict):
#         """
#         Like flatten_metrics, but replaces 'class_{i}' with real label names.
#         """
#         for name, value in metrics.items():
#             # pull numpy array or scalar
#             if isinstance(value, torch.Tensor):
#                 arr = value.detach().cpu().numpy()
#             else:
#                 arr = value

#             # array-like → unroll per index
#             if hasattr(arr, "__len__") and not isinstance(arr, (str, bytes)):
#                 for i, v in enumerate(arr):
#                     # pick the right map
#                     if name in ("sean_f1", "santiago_f1"):
#                         label = self.idx_to_class.get(i, f"class_{i}")
#                     elif name == "base_f1":
#                         label = self.idx_to_base.get(i, f"{i+1}")
#                     elif name == "subclass_f1":
#                         label = self.idx_to_subclass.get(i, f"sub{i}")
#                     else:
#                         label = f"class_{i}"
#                     out_dict[f"{prefix}_{name}_{label}"] = float(v)
#             else:
#                 # scalar
#                 out_dict[f"{prefix}_{name}"] = float(arr)
