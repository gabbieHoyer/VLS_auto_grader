# Dataset and Augmentation Overview (`src/data/`)

This module defines how video data is loaded, processed, and augmented for both **supervised training** and **self-supervised pretraining** in the `VLS_auto_grader` project.

---

## 📦 `MultiGraderDataset`

This dataset class serves as the foundation for supervised training. It supports various labeling modes:

* **Single-Grader** or **Multi-Grader** labels
* **Hierarchical** (base + subclass) classification
* **Non-Hierarchical** (flat) classification
* **Simplified Base-Only** classification

Each video is loaded and subsampled to a fixed number of frames (default: 16). Labels are parsed and structured differently depending on the experimental configuration.

### Label Modes:

| Mode             | Configuration Flag               | Output Structure                                                            |
| ---------------- | -------------------------------- | --------------------------------------------------------------------------- |
| Simplified Base  | `simplified_base: true`          | Class index from base class only                                            |
| Hierarchical     | `num_subclasses > 0`             | Dictionary with `base_label`, `subclass_label`, and `valid_subclasses` mask |
| Non-Hierarchical | `num_classes > num_base_classes` | Class index from flattened class set                                        |

The dataset dynamically handles rater label combinations (e.g., `Sean_Review`, `Santiago_Review`) and uses label aggregation logic for multi-rater inputs.

---

## 📦 `PretrainingDataset`

This class wraps `MultiGraderDataset` for use in self-supervised learning (SSL) tasks. It supports:

* **Contrastive** / **MoCo**: Requires two transformed views of the same video
* **MAE**: Applies temporal masking with consistency controls across frames

### MAE Masking Strategies:

* `full`: A single mask is applied to all frames
* `partial`: Mask is updated every `change_interval` frames
* `none`: Mask applied independently per frame

Augmentations are replayed with `ReplayCompose` to maintain temporal consistency.

---

## 🧪 Augmentation Pipeline

### 🔧 `get_transforms()` (used in supervised training)

Applies frame-level transformations consistently across the temporal dimension:

* Random resized crop
* Horizontal flip
* Rotation
* Color jitter
* Normalization

### 🔧 `get_ssl_transforms(pretrain_method)`

* For `contrastive` and `moco`: Applies stronger augmentations to produce diverse views
* For `mae`: Uses minimal spatial and color transformations to preserve structure

All augmentations are temporally **consistent** via `ReplayCompose`, ensuring each frame in the sequence receives the same spatial and color transformation.

---

## 🧠 Design Highlights

* Augmentation pipelines for SSL tasks attempt to reflect **temporal coherence** while still promoting sufficient diversity (contrastive) or controlled corruption (MAE).
* Supervised datasets support dynamic label aggregation and format changes based on experiment setup.
* Pretraining setups are **architecture-agnostic** and designed to generalize across future 3D video models.

---

## 💡 Extending the Dataset Pipeline

You can modify or extend the following components for new tasks:

* `augmentations.py`: Add spatial, temporal, or domain-specific augmentations (e.g., motion blur, grayscale jitter)
* `apply_masking()`: Add new temporal masking strategies
* `MultiGraderDataset`: Handle new label schemas or metadata fields (e.g., anatomical region, video duration bins)

---

## References

* `src/data/dataset.py`: Core dataset classes
* `src/data/augmentations.py`: Augmentation logic with temporal consistency
* `src/ssl/ssl_helpers.py`: MAE-style masking utilities
* `src/training/train.py`: Integration of dataset logic into training pipeline
