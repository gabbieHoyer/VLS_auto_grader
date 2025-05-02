# src/utils/wandb_utils.py
import wandb

def init_wandb_run(cfg, run_id, stage, tags=None):
    """
    Initialize a wandb run for pretrain or finetune, with optional tags.

    Args:
      cfg     : full config dict
      run_id  : str timestamp or other unique id
      stage   : "pretrain" or "finetune"
      tags    : list of str; if None a default set is constructed

    Returns:
      wandb.Run
    """
    exp_name = cfg['experiment']['name']
    project  = cfg['output_configuration']['task_name']

    # build default tags if none passed
    if tags is None:
        if stage == 'pretrain':
            tags = ['pretrain', exp_name, cfg['training']['pretrain_method']]
        else:
            ssl = 'ssl' if cfg['training']['ssl_pretrain'] else 'baseline'
            tags = ['train', exp_name, ssl]

    # decide job_type and run name
    if stage == 'pretrain':
        job_type = 'pretrain'
        name     = f"{run_id}_pretrain"
    else:
        if cfg['training']['ssl_pretrain']:
            job_type = 'finetune_ssl'
            name     = f"{run_id}_finetune_ssl"
        else:
            job_type = 'finetune_baseline'
            name     = f"{run_id}_finetune_baseline"

    run = wandb.init(
        project   = project,
        config    = cfg['training'],
        group     = exp_name,
        job_type  = job_type,
        name      = name,
        tags      = tags,
        resume    = False
    )
    return run



# self.model_save_path, self.run_id = self.setup_experiment_environment()

# def setup_experiment_environment(self):
# model_save_path = self.run_path
# run_id = datetime.now().strftime("%Y%m%d-%H%M")
# if self.cfg['output_configuration'].get('use_wandb') and GPUSetup.is_main_process():
#     import wandb
#     wandb.init(
#         project=self.cfg['output_configuration']['task_name'],
#         config=self.cfg['training'],
#         name=run_id,
#         tags=['train', self.cfg['experiment']['name'], 'supervised']
#     )
# return model_save_path, run_id