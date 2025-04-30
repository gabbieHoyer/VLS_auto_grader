# src/utils/experiment_setup.py
import os
import pyrootutils

def get_project_root():
    """Find the project root directory based on .git indicator."""
    return pyrootutils.setup_root(
        search_from=__file__,
        indicator=[".git"],
        pythonpath=True,
        dotenv=True,
    )

# def generate_group_name(cfg):
#     """Generate a dynamic group name based on config settings."""
#     ssl_pretrain = cfg['training']['ssl_pretrain']
#     model_name = cfg['training']['model_name']
#     return f"{model_name}_ssl_{'pretrain' if ssl_pretrain else 'supervised'}"

def generate_group_name(cfg):
    """
    Hybrid group-name generator:
    1) If the user has set experiment.name in the config, use that verbatim.
    2) Otherwise build one from model, hierarchy, graders, and SSL flags.
    """
    # 1) fallback to explicit experiment.name
    exp_name = cfg.get('experiment', {}).get('name')
    if exp_name:
        return exp_name

    # 2) dynamic construction
    model = cfg['training']['model_name']
    ssl   = 'pretrain' if cfg['training']['ssl_pretrain'] else 'supervised'

    label_cols = cfg['training']['datamodule'].get('label_cols', [])
    grade      = 'multigrader'  if len(label_cols) > 1 else 'singlegrader'

    num_subs   = cfg['training'].get('num_subclasses', 0)
    hier       = 'hierarchical'    if num_subs > 0 else 'nonhierarchical'

    return f"{model}_{hier}_{grade}_{ssl}"


def determine_run_directory(base_dir, task_name, cfg, group_name=None):
    """
    Determines the next run directory for storing experiment data.

    Args:
        base_dir (str): Base directory (e.g., work_dir/finetuning).
        task_name (str): Task name (e.g., VLS-3D-Grading).
        cfg (dict): Configuration dictionary for dynamic group name generation.
        group_name (str, optional): Explicit group name; if None, generated dynamically.

    Returns:
        str: Full path to the run directory (e.g., work_dir/finetuning/VLS-3D-Grading/i3d_ssl_pretrain/Run_1).
    """
    if group_name is None:
        group_name = generate_group_name(cfg)
    
    base_path = os.path.join(base_dir, task_name, group_name)
    os.makedirs(base_path, exist_ok=True)
    
    # Filter for directories that start with 'Run_' and are followed by an integer
    existing_runs = []
    for d in os.listdir(base_path):
        if d.startswith('Run_') and os.path.isdir(os.path.join(base_path, d)):
            parts = d.split('_')
            if len(parts) == 2 and parts[1].isdigit():  # Check if there is a number after 'Run_'
                existing_runs.append(d)
    
    if existing_runs:
        # Sort by the integer value of the part after 'Run_'
        existing_runs.sort(key=lambda x: int(x.split('_')[-1]))
        last_run_num = int(existing_runs[-1].split('_')[-1])
        next_run_num = last_run_num + 1
    else:
        next_run_num = 1
    
    run_directory = f'Run_{next_run_num}'
    full_run_path = os.path.join(base_path, run_directory)
    os.makedirs(full_run_path, exist_ok=True)
    
    return full_run_path