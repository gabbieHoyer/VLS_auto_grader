# Experiment Configurations for VLS Auto Grader

This directory contains YAML configuration files used to train and evaluate models for the `VLS_auto_grader` pipeline. Each file defines the setup for a specific experiment varying by:

* **Grader setting**: Single-grader vs. Multi-grader
* **Classification type**: Hierarchical, Non-Hierarchical, or Simplified Base Class
* **Pretraining strategy**: Supervised only vs. pretraining with MoCo, MAE, or Contrastive learning

## Naming Convention

`{classification}_{grader_setting}_{ssl_flag}.yaml`

* `classification` ∈ {`hierarchical`, `nonhierarchical`, `baseline` (for simplified)}
* `grader_setting` ∈ {`singlegrader`, `multigrader`}
* `ssl_flag` ∈ {`ssl`, `nossl`}

---

## Summary of Experiments

### 1. **Simplified (Baseline) Classifier**

**Example**: `baseline_singlegrader_nossl.yaml`

* Trains on base class labels only (`simplified_base: true`)
* Ignores subclass distinctions
* Single-rater label set (e.g., `Sean_Review`)
* No pretraining (`ssl_pretrain: false`)

### 2. **Non-Hierarchical Classification**

**Example**: `nonhierarchical_singlegrader_nossl.yaml`

* Uses all class/subclass labels as independent classes
* Only `Sean_Review` grader label is used
* No pretraining

Relevant fields:

```yaml
training:
  simplified_base: false
  num_base_classes: 8
  num_subclasses: 0
  num_classes: 8
  datamodule:
    label_cols: ['Sean_Review']
  ssl_pretrain: false
```

### 3. **Non-Hierarchical + Multi-Grader + SSL**

**Example**: `nonhierarchical_multigrader_ssl.yaml`

* Uses full class set as independent classes
* Multiple graders (`Sean_Review`, `Santiago_Review`)
* Self-supervised pretraining enabled (e.g., contrastive)

Relevant fields:

```yaml
training:
  ssl_pretrain: true
  pretrain_method: 'contrastive'
```

### 4. **Hierarchical Classification**

**Example**: `hierarchical_singlegrader_ssl.yaml`

* Uses structured base + subclass classification heads
* Subclass loss scaled with `subclass_loss_weight`
* Supports MAE or MoCo pretraining

Relevant fields:

```yaml
training:
  num_base_classes: 4
  num_subclasses: 5
  subclass_loss_weight: 0.5
  ssl_pretrain: true
  pretrain_method: 'mae'
```

### 5. **Hierarchical + Multi-Grader**

**Example**: `hierarchical_multigrader_nossl.yaml`

* Multi-rater aggregation for base/subclass structure
* No pretraining

```yaml
training:
  datamodule:
    label_cols: ['Sean_Review', 'Santiago_Review']
  ssl_pretrain: false
```

---

## Configuration Field Highlights

### Model Type

* Specify architecture via:

```yaml
training:
  model_name: 'i3d'  # Other architectures can be added
```

### Dataset CSV

* Specify the dataset file and split assignments:

```yaml
paths:
  data_csv: data/video_splits.csv
```

### Pretraining Options

* `ssl_pretrain`: `true` or `false`
* `pretrain_method`: `mae`, `moco`, `contrastive`
* These pretraining options are agnostic to the downstream classification setup (hierarchical/non-hierarchical, single/multi-grader, or simplified base).
* For MAE-specific training:

```yaml
  patch_size: 16
  mask_ratio: 0.5
  end_mask_ratio: 0.75
  temporal_consistency: 'full'
  change_interval: 5
```

### Label Structure Options

* `simplified_base`: set `true` to use base class labels only
* `num_base_classes`, `num_subclasses`, `num_classes`: control architecture and metric computation

### Grader Control

* `label_cols`: one or both rater columns (e.g., `['Sean_Review']`, `['Sean_Review', 'Santiago_Review']`)

---

## Notes

* Early stopping, label smoothing, and gradient accumulation are adjustable per config.
* `use_wandb` can be toggled for logging experiment runs.
* All experiments write to `work_dir/runs/{experiment_type}/{model_type}/...` based on the config's metadata.

If adding new experiment types, follow the existing pattern for naming and field usage to ensure compatibility with the training engine and dataset structure.
