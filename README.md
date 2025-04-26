# VLS Auto Grader - Video Classification for Intubation Procedures

The `VLS_auto_grader` project implements a deep learning pipeline for classifying 3D temporal videos of medical procedures using 2025 approaches, including 3D CNNs (I3D) and Video Vision Transformers (ViViT).

## Project Goals

- **Automated Data Cleaning**: Remove irrelevant frames and ensure privacy.
- **Modular Design**: Clear separation of preprocessing, modeling, and UI components.

## Repository Structure

```text
root/
├── config/
│   └── config.yaml           # Configuration file
├── src/
│   ├── train.py              # Training script
│   ├── eval.py               # Evaluation script
│   ├── dataset.py            # Dataset and DataLoader
│   ├── model.py              # Model definitions
│   ├── metrics.py            # Evaluation metrics
│   ├── utils/
│   │   ├── __init__.py       # Package init
│   │   ├── augmentations.py  # Data augmentations
│   │   ├── io.py             # Video loading utilities
│   │   ├── logger.py         # Logging utilities
│   │   ├── ssl_utils.py      # SSL utilities
│   │   └── config_loader.py  # Config loading utilities
├── data/
│   └── video_data.csv        # CSV with video paths and labels
├── work_dir/
│   ├── experiment_runs/      # Output directory for experiment runs, checkpoints
│   └── model_weights/        # Model weights for pretraining/transfer learning
└── README.md
```

For detailed instructions and examples, see the README in each major subfolder:

- [Dataset Preprocessing](src/dataset_preprocessing/README.md)
- [Face/Person Detection](src/face_detection/README.md)


## Quickstart

1. **Clone & install**:
   ```bash
   git clone https://github.com/gabbieHoyer/VLS_auto_grader.git
   cd VLS_auto_grader
   ```
    Install dependencies (choose one option):

    **Option 1: Using Conda (recommended for cross-platform compatibility and GPU support)**
    Install Miniconda if not already installed, then:
    ```bash
    conda env create -f environment.yml
    conda activate my_project_env
    ```

    **Option 2: Using pip and a virtual environment**
    Ensure Python 3.9+ is installed, then:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

2. Prepare your `data/video_data.csv` with columns:

- `video_path`: Full path to video files  
- `label`: Classification label (integer or string)  
- `split`: Train/val/test assignment

3. **Prepare configuration**:
   ```bash
   cp config/config.yaml.example config/config.yaml
   ```
   Edit the copied YAMLs under `config/` to set your paths and parameters.

Create a config.yaml file in root/config/ (see example below).

### Example config.yaml

```yaml
paths:
  data_csv: data/video_data.csv
  log_dir: logs/
  checkpoint_dir: checkpoints/
training:
  model_name: i3d
  num_classes: 3
  batch_size: 4
  epochs: 50
  lr: 1e-4
  ssl_pretrain: false
  ssl_epochs: 20
  ssl_lr: 1e-3
  num_workers: 4
evaluation:
  split: val
  save_attention: false
```

## Usage

### Training

Run with default config values:
```bash
python src/train.py
```

Override specific parameters:
```bash
python src/train.py --data_csv data/video_data.csv --model_name i3d --num_classes 3 --batch_size 4 --epochs 50 --lr 1e-4
```
or
```bash
python src/train.py --config_file config.yaml --model_name vivit --num_classes 3 --ssl_pretrain
```

### Evaluation

Run with default config values:
```bash
python src/eval.py --checkpoint_path checkpoints/checkpoint_epoch_50.pth
```

Override specific parameters:

```bash
python src/eval.py --data_csv data/video_data.csv --model_name i3d --num_classes 3 --split test --checkpoint_path checkpoints/checkpoint_epoch_50.pth
```
or
```bash
python src/eval.py --config_file config.yaml --split test --checkpoint_path checkpoints/checkpoint_epoch_50.pth
```

## Notes

- The codebase supports **I3D** and **ViViT** models.
- Parameters in `config.yaml` can be overridden via command-line arguments.
- **ViViT** includes a placeholder for attention visualization.
- Extend augmentations in `augmentations.py` for medical-specific needs.
- For distributed training, integrate `GPUSetup` in `train.py` and `eval.py`.


## Requirements

- Python 3.8+
- PyTorch 1.12+
- torchvision
- albumentations
- pandas
- numpy
- opencv-python
- scikit-learn
-tqdm
- pyyaml


## Next Steps

After setup and preprocessing, follow subfolder READMEs to:

- Train/infer the **procedural frame classification** model.
- Train/infer the **procedural frame classification** model.

## Data Usage

Details on the datasets used in this codebase—frame classification images, face detection annotations, and anonymized trimmed videos—are documented in [data/README.md](data/README.md).

## Future Work

- **Performance Optimization**: Speed up frame and video processing for large datasets.
- **Model Improvements**: Enhance accuracy for both procedural classification and face detection.

## License

This project is licensed under the MIT License. (placeholder)

## Reference

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

- **Gabrielle Hoyer**  
  - UCSF / UC Berkeley  
  - [gabbie.hoyer@ucsf.edu](mailto:gabbie.hoyer@ucsf.edu)  
  - [gabrielle_hoyer@berkeley.edu](mailto:gabrielle_hoyer@berkeley.edu)
  
For questions or collaboration, feel free to reach out via email.

