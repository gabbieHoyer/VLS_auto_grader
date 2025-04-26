from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np

def compute_metrics(labels, preds, num_classes):
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(labels, preds)
    
    # F1 Score
    metrics['f1_score'] = f1_score(labels, preds, average='macro')
    
    # AUC-ROC (for binary or multi-class)
    if num_classes == 2:
        metrics['auc_roc'] = roc_auc_score(labels, preds)
    else:
        # For multi-class, compute one-vs-rest AUC
        try:
            metrics['auc_roc'] = roc_auc_score(labels, preds, multi_class='ovr')
        except:
            metrics['auc_roc'] = float('nan')  # Handle cases where AUC is not applicable
    
    return metrics