# VLS Auto Grader – Video Classification for Intubation Procedures

The `VLS_auto_grader` project tackles the intricate task of classifying 3D temporal videos of video laryngoscope (VLS) intubation procedures. Unlike static image classification, this task demands understanding both spatial and temporal dynamics. It involves recognizing sequences of actions (e.g., laryngoscope insertion, vocal cord visualization) and the 3D spatial configurations of medical instruments and patient anatomy, resembling action recognition and 3D pose estimation in computer vision.

The medical context amplifies this complexity:

* **Procedural Variability**: Intubation procedures vary in technique and difficulty, requiring models to discern subtle differences.
* **3D Depth Information**: The 3D nature of videos introduces depth ambiguities, necessitating precise spatial analysis.
* **Clinical Relevance**: Classifications align with clinical standards, such as the Video Classification of Intubation (VCI) score (VCI Score), ensuring practical utility.

## Labeling System

The classification label set assesses **Overall Procedure Difficulty** based on procedural success, equipment motion, anatomical displacement, and trauma, informed by metrics like time, Cormack–Lehane (CL) grade, blood/erythema, and number of attempts. Classes include:

* **1**: Easy tracheal access (successful, single smooth motion, minimal displacement).
* **2**: Moderately difficult access (successful, multiple reversals/redirections, moderate displacement, no trauma). Subclasses **2b**, **2c**, **2d** indicate specific challenges.
* **3**: Severely difficult access (successful, multiple reversals, severe displacement, visible trauma). Subclasses **3b**, **3c** indicate specific severe challenges.
* **4**: Impossible access (unsuccessful placement of endotracheal tube). Subclasses **4a–4d** detail failure points.

### Rater-Specific Labels

* **Santiago\_Review** includes: `1`, `2`, `2b`, `2c`, `2d`, `3`, `3b`, `4b`
* **Sean\_Review** includes: `1`, `2`, `2b`, `2c`, `3`, `3b`, `3c`, `4b`, `x` (ETT check)

## Project Goals

* **Training Models for Procedural Grading**: Develop classifiers to learn spatial and temporal representations for assessing difficulty and technique.
* **Multi-Rater Support**: Experiment with single- or multi-rater annotations to address inter-rater variability.
* **Flexible Classification Schema**: Test hierarchical, non-hierarchical, and simplified labeling schemes.
* **Pretraining Strategies**: Use self-supervised learning (e.g., MAE, MoCo, contrastive) on unlabeled VLS videos for low-data regimes.
* **Transfer Learning**: Pretrain on larger surgical video datasets, then finetune on VLS data.
* **Modular Pipeline**: Separate modules for preprocessing, augmentation, training, evaluation, visualization, and logging.
* **Action Recognition–Like Modeling**: Model motion, temporal cues, and spatial context—akin to action recognition rather than static classification.

## Repository Structure

```text
root/
├── config/                    # Experiment configuration YAMLs
│   └── dev/                  # Development configs (SSL, hierarchical, etc.)
├── data/                     # Video splits and related metadata
│   ├── video_splits.csv
│   └── video_splits_corrected.csv
├── notebooks/                # Data and augmentation exploration notebooks
│   ├── analyze_sean_review.py
│   ├── augmentation_qc_initial_design.ipynb
│   ├── augmentation_qc_temporal_design.ipynb
│   └── dataset_exploration.ipynb
├── src/                      # Source code base
│   ├── __init__.py
│   ├── data/                 # Data pipeline
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── augmentations.py
│   │   └── summarize_label_distribution.py
│   ├── models/               # Model definitions and metrics
│   │   ├── __init__.py
│   │   ├── model.py
│   │   └── metrics.py
│   ├── ssl/                  # SSL losses and helpers
│   │   ├── __init__.py
│   │   ├── ssl_losses.py
│   │   └── ssl_helpers.py
│   ├── training/             # Training and evaluation scripts
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── eval.py
│   │   └── engine.py
│   └── utils/                # Utilities
│       ├── __init__.py
│       ├── checkpointing.py
│       ├── config_loader.py
│       ├── gpu_setup.py
│       ├── logger.py
│       ├── paths.py
│       ├── project.py
│       ├── visualization.py
│       └── wandb_utils.py
├── work_dir/                 # Experiment outputs
│   ├── runs/                 # Organized by experiment type/model
│   │   ├── baseline_train/i3d/
│   │   ├── pretrain_and_finetune/
│   │   └── data_summary/
└── README.md
```

## Entry Points

Train and evaluate models using the following:

```bash
python -m src.training.train --config config/dev/{config_name}.yaml
```

```bash
python -m src.training.eval --config config/dev/{config_name}.yaml
```

## Output Folder Design (under `work_dir/runs/`)

Each experiment run is organized as:

```text
work_dir/runs/{experiment_type}/{model_type}/{experiment_name}/Run_{i}/
├── checkpoints/        # Stores model weights (e.g., best_model.pth, ssl_pretrained.pth)
├── logs/               # Logs for training and evaluation
│   ├── train.log
│   └── eval.log
├── figures/            # Visualizations (confusion matrices, loss curves, attention maps)
├── summaries/          # Text files summarizing the experiment setup and results
```

## Quickstart

1. **Clone & install**:

```bash
git clone https://github.com/gabbieHoyer/VLS_auto_grader.git
cd VLS_auto_grader
```

2. **Install dependencies**:

**Option 1: Using Conda (recommended for cross-platform compatibility and GPU support)**
  Install Miniconda if not already installed, then:
```bash
conda env create -f environment.yml
conda activate vls_env
```

**Option 2: Using pip and a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Prepare your CSV**:

Ensure your `data/video_splits.csv` has:

* `video_path`: Full path to video files
* `label`: Classification label
* `split`: Train/val/test

4. **Edit config YAML**:

Example: `config/hierarchical_singlegrader_ssl.yaml`

```yaml
experiment:
  name: VLS_3D_Grading_Hierarchical_SingleGrader_SSL
  description: "Hierarchical single-grader training with SSL pretraining."

paths:
  data_csv: data/video_splits_corrected.csv
  log_dir: logs
  checkpoint_dir: checkpoints
  figures_dir: figures
  summaries_dir: summaries

output_configuration:
  work_dir: work_dir/runs
  task_name: vls_grading
  use_wandb: true
  checkpoint_interval: 5
  logging_level: INFO

training:
  datamodule:
    video_col: 'Processed_video_path'
    label_cols: ['Sean_Review']
  model_name: i3d
  num_base_classes: 4
  num_subclasses: 5
  num_classes: 8
  batch_size: 16
  num_workers: 1

  ssl_lr: 0.001
  ssl_epochs: 20
  ssl_pretrain: true
  pretrain_method: moco
  subclass_loss_weight: 0.5

  mask_ratio: 0.5
  end_mask_ratio: 0.75
  patch_size: 16
  temporal_consistency: full
  change_interval: 5

  lr: 0.0001
  epochs: 50

  optimizer:
    weight_decay: 0.005
  loss:
    label_smoothing: 0.05

  module:
    use_amp: true
    clip_grad: 1.0
    grad_accum: 4
    early_stopping:
      enabled: true
      patience: 10
      min_delta: 0.0001

evaluation:
  split: test
  checkpoint_path: checkpoints/best_model.pth
  save_attention: false

distributed: true
SEED: 42
```

## Usage

### Training

```bash
python -m src.training.train --config config/hierarchical_singlegrader_ssl.yaml
```

### Evaluation

```bash
python -m src.training.eval --config config/hierarchical_singlegrader_ssl.yaml --checkpoint_path path/to/checkpoint.pth
```

## Notes

* Configurable to support **hierarchical**, **non-hierarchical**, and **simplified** class structures.
* Supports **I3D**, **ViViT**, and **self-supervised learning (SSL)** via contrastive, MoCo, and MAE.
* Extend `augmentations.py` for medical-specific transforms.
* Use `summarize_label_distribution.py` to verify label balance.

## Requirements

* Python 3.8+
* PyTorch 1.12+
* torchvision
* albumentations
* pandas
* numpy
* opencv-python
* scikit-learn
* tqdm
* pyyaml

## Citation

```bibtex
@misc{hoyer2025VLSAutoGrader,
  author       = {Hoyer, Gabrielle and Runnels, Sean},
  title        = {VLSAutoGrader},
  year         = {2025},
  howpublished = {Computer software},
  version      = {1.0.0},
  note         = {Available at \url{https://github.com/gabbieHoyer/VLS_auto_grader}},
}
```

## Contact

**Gabrielle Hoyer**
UCSF / UC Berkeley
[gabbie.hoyer@ucsf.edu](mailto:gabbie.hoyer@ucsf.edu)
[gabrielle\_hoyer@berkeley.edu](mailto:gabrielle_hoyer@berkeley.edu)

For questions or collaboration, feel free to reach out.



