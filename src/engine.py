import os
import torch
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime
import logging
import numpy as np

from utils import GPUSetup, log_info, wandb_log
from utils.metrics import compute_metrics
from utils.visualization import plot_confusion_matrix, save_attention_maps, plot_loss_curves
from utils.checkpointing import log_and_checkpoint, final_checkpoint_conversion

# from utils.ssl_losses import ReconstructionLoss, ContrastiveLoss
from utils.ssl_losses import SimCLRContrastiveLoss, MoCoLoss, ReconstructionLoss

logger = logging.getLogger(__name__)

# Mapping for reconstructing labels in hierarchical case
BASE_CLASSES = {0: '1', 1: '2', 2: '3', 3: '4'}
SUBCLASSES = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: ''}


class TrainingEngine:
    def __init__(self, model, optimizer, scheduler, criterion, train_loader, val_loader, cfg, run_path, device, start_epoch=0):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.run_path = run_path
        self.device = device
        self.start_epoch = start_epoch
        self.use_amp = cfg['training']['module'].get('use_amp', False)
        self.scaler = GradScaler(enabled=self.use_amp)
        self.clip_grad = cfg['training']['module'].get('clip_grad', 1.0)
        self.grad_accum = cfg['training']['module'].get('grad_accum', 1)
        self.early_stopping = cfg['training']['module'].get('early_stopping', {})
        self.best_val_metric = 0.0
        self.no_improve_epochs = 0
        self.model_save_path, self.run_id = self.setup_experiment_environment()
        self.hierarchical_mode = self._check_hierarchical_mode()
        self.single_grader = self._check_single_grader()

    def setup_experiment_environment(self):
        model_save_path = self.run_path
        run_id = datetime.now().strftime("%Y%m%d-%H%M")
        if self.cfg['output_configuration'].get('use_wandb') and GPUSetup.is_main_process():
            import wandb
            wandb.init(
                project=self.cfg['output_configuration']['task_name'],
                config=self.cfg['training'],
                name=run_id,
                tags=['train', self.cfg['experiment']['name'], 'supervised']
            )
        return model_save_path, run_id

    def _check_hierarchical_mode(self):
        if self.train_loader is None:
            return False
        batch = next(iter(self.train_loader))
        return isinstance(batch, dict) and 'base_label' in batch

    def _check_single_grader(self):
        if self.train_loader is None:
            return False
        batch = next(iter(self.train_loader))
        if self.hierarchical_mode:
            return False  # In hierarchical mode, labels are aggregated, so single_grader doesn't apply
        return len(batch[1]) == 1 or (len(batch[1]) == 2 and torch.equal(batch[1][0], batch[1][1]))

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        all_base_preds, all_subclass_preds, all_base_labels, all_subclass_labels = [], [], [], []
        batch_count = 0

        for batch_idx, batch in enumerate(self.train_loader):
            if self.hierarchical_mode:
                videos = batch['video'].to(self.device)
                base_labels = batch['base_label'].to(self.device)
                subclass_labels = batch['subclass_label'].to(self.device)

                with autocast(enabled=self.use_amp):
                    outputs = self.model(videos)
                    loss = self.criterion(outputs, batch)
                    base_preds = torch.argmax(outputs[0], dim=1)
                    subclass_preds = torch.argmax(outputs[1], dim=1)
                    all_base_preds.append(base_preds.cpu())
                    all_subclass_preds.append(subclass_preds.cpu())
                    all_base_labels.append(base_labels.cpu())
                    all_subclass_labels.append(subclass_labels.cpu())
            else:
                videos, labels = batch
                videos = videos.to(self.device)
                labels = [label.to(self.device) for label in labels]
                sean_labels = labels[0]
                with autocast(enabled=self.use_amp):
                    outputs = self.model(videos)
                    loss = self.criterion(outputs, sean_labels)
                    preds = torch.argmax(outputs, dim=1)
                    all_base_preds.append(preds.cpu())
                    all_base_labels.append(sean_labels.cpu())
                    if not self.single_grader:
                        santiago_labels = labels[1]
                        all_subclass_labels.append(santiago_labels.cpu())
                        all_subclass_preds.append(preds.cpu())

            self.scaler.scale(loss / self.grad_accum).backward()
            if (batch_idx + 1) % self.grad_accum == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                if self.scheduler:
                    self.scheduler.step()

            total_loss += loss.item() * self.grad_accum
            batch_count += 1
            if batch_idx % 10 == 0 and GPUSetup.is_main_process():
                log_info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        if GPUSetup.is_distributed():
            total_loss_tensor = torch.tensor(total_loss, device=self.device)
            torch.distributed.all_reduce(total_loss_tensor)
            total_loss = total_loss_tensor.item() / torch.distributed.get_world_size()
            batch_count_tensor = torch.tensor(batch_count, device=self.device)
            torch.distributed.all_reduce(batch_count_tensor)
            batch_count = batch_count_tensor.item() / torch.distributed.get_world_size()

        metrics = {}
        if all_base_preds:
            all_base_preds = torch.cat(all_base_preds)
            all_base_labels = torch.cat(all_base_labels)
            if self.hierarchical_mode:
                all_subclass_preds = torch.cat(all_subclass_preds)
                all_subclass_labels = torch.cat(all_subclass_labels)
                base_metrics = compute_metrics(all_base_labels, all_base_preds, self.cfg['training']['num_base_classes'])
                subclass_metrics = compute_metrics(all_subclass_labels, all_subclass_preds, self.cfg['training']['num_subclasses'])
                metrics = {
                    'base_accuracy': base_metrics['accuracy'],
                    'base_f1': base_metrics['f1_per_class'],
                    'subclass_accuracy': subclass_metrics['accuracy'],
                    'subclass_f1': subclass_metrics['f1_per_class']
                }
            else:
                sean_metrics = compute_metrics(all_base_labels, all_base_preds, self.cfg['training']['num_classes'])
                metrics = {
                    'sean_accuracy': sean_metrics['accuracy'],
                    'sean_f1': sean_metrics['f1_per_class']
                }
                if not self.single_grader:
                    all_subclass_preds = torch.cat(all_subclass_preds)
                    all_subclass_labels = torch.cat(all_subclass_labels)
                    santiago_metrics = compute_metrics(all_subclass_labels, all_subclass_preds, self.cfg['training']['num_classes'])
                    metrics.update({
                        'santiago_accuracy': santiago_metrics['accuracy'],
                        'santiago_f1': santiago_metrics['f1_per_class']
                    })

        return total_loss / batch_count, metrics

    def validate(self):
        if self.val_loader is None:
            return 0, {}, None, None, None, None

        self.model.eval()
        total_loss = 0
        all_base_preds, all_subclass_preds, all_base_labels, all_subclass_labels = [], [], [], []
        batch_count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                if self.hierarchical_mode:
                    videos = batch['video'].to(self.device)
                    base_labels = batch['base_label'].to(self.device)
                    subclass_labels = batch['subclass_label'].to(self.device)
                    with autocast(enabled=self.use_amp):
                        outputs = self.model(videos)
                        loss = self.criterion(outputs, batch)
                        base_preds = torch.argmax(outputs[0], dim=1)
                        subclass_preds = torch.argmax(outputs[1], dim=1)
                        all_base_preds.append(base_preds.cpu())
                        all_subclass_preds.append(subclass_preds.cpu())
                        all_base_labels.append(base_labels.cpu())
                        all_subclass_labels.append(subclass_labels.cpu())
                else:
                    videos, labels = batch
                    videos = videos.to(self.device)
                    labels = [label.to(self.device) for label in labels]
                    sean_labels = labels[0]
                    with autocast(enabled=self.use_amp):
                        outputs = self.model(videos)
                        loss = self.criterion(outputs, sean_labels)
                        preds = torch.argmax(outputs, dim=1)
                        all_base_preds.append(preds.cpu())
                        all_base_labels.append(sean_labels.cpu())
                        if not self.single_grader:
                            santiago_labels = labels[1]
                            loss_santiago = self.criterion(outputs, santiago_labels)
                            loss = (loss + loss_santiago) / 2
                            all_subclass_preds.append(preds.cpu())
                            all_subclass_labels.append(santiago_labels.cpu())
                        else:
                            all_subclass_preds.append(preds.cpu())
                            all_subclass_labels.append(sean_labels.cpu())  # Duplicate for compatibility

                total_loss += loss.item()
                batch_count += 1

        if GPUSetup.is_distributed():
            total_loss_tensor = torch.tensor(total_loss, device=self.device)
            torch.distributed.all_reduce(total_loss_tensor)
            total_loss = total_loss_tensor.item() / torch.distributed.get_world_size()
            batch_count_tensor = torch.tensor(batch_count, device=self.device)
            torch.distributed.all_reduce(batch_count_tensor)
            batch_count = batch_count_tensor.item() / torch.distributed.get_world_size()

        all_base_preds = torch.cat(all_base_preds)
        all_base_labels = torch.cat(all_base_labels)
        all_subclass_preds = torch.cat(all_subclass_preds)
        all_subclass_labels = torch.cat(all_subclass_labels)

        metrics = {}
        if self.hierarchical_mode:
            base_metrics = compute_metrics(all_base_labels, all_base_preds, self.cfg['training']['num_base_classes'])
            subclass_metrics = compute_metrics(all_subclass_labels, all_subclass_preds, self.cfg['training']['num_subclasses'])
            metrics = {
                'base_accuracy': base_metrics['accuracy'],
                'base_f1': base_metrics['f1_per_class'],
                'subclass_accuracy': subclass_metrics['accuracy'],
                'subclass_f1': subclass_metrics['f1_per_class']
            }
        else:
            sean_metrics = compute_metrics(all_base_labels, all_base_preds, self.cfg['training']['num_classes'])
            metrics = {
                'sean_accuracy': sean_metrics['accuracy'],
                'sean_f1': sean_metrics['f1_per_class']
            }
            if not self.single_grader:
                santiago_metrics = compute_metrics(all_subclass_labels, all_subclass_preds, self.cfg['training']['num_classes'])
                metrics.update({
                    'santiago_accuracy': santiago_metrics['accuracy'],
                    'santiago_f1': santiago_metrics['f1_per_class']
                })

        return total_loss / batch_count, metrics, all_base_preds, all_base_labels, all_subclass_preds, all_subclass_labels

    def check_early_stopping(self, val_metric):
        if not self.early_stopping.get('enabled', False):
            return False
        patience = self.early_stopping.get('patience', 10)
        min_delta = self.early_stopping.get('min_delta', 0.0001)
        should_stop = False
        if val_metric > self.best_val_metric + min_delta:
            self.no_improve_epochs = 0
            self.best_val_metric = val_metric
        else:
            self.no_improve_epochs += 1
            if self.no_improve_epochs >= patience:
                should_stop = True

        if GPUSetup.is_distributed():
            stop_tensor = torch.tensor([1 if should_stop else 0], dtype=torch.int, device=self.device)
            torch.distributed.broadcast(stop_tensor, src=0)
            should_stop = stop_tensor.item() == 1

        if should_stop:
            log_info(f"Early stopping triggered after {self.no_improve_epochs} epochs")
        return should_stop

    def train(self, epochs):
        losses = {'train': [], 'val': []}
        latest_val_base_preds, latest_val_base_labels = None, None
        latest_val_subclass_preds, latest_val_subclass_labels = None, None

        for epoch in range(self.start_epoch, epochs):
            if GPUSetup.is_distributed() and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            train_loss, train_metrics = self.train_epoch(epoch)
            val_loss, val_metrics, val_base_preds, val_base_labels, val_subclass_preds, val_subclass_labels = self.validate()

            if val_base_preds is not None and val_base_labels is not None:
                latest_val_base_preds, latest_val_base_labels = val_base_preds, val_base_labels
                latest_val_subclass_preds, latest_val_subclass_labels = val_subclass_preds, val_subclass_labels

            if GPUSetup.is_main_process():
                log_info(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                log_info(f"Train Metrics: {train_metrics}")
                log_info(f"Val Metrics: {val_metrics}")
                losses['train'].append({'epoch': epoch, 'loss': train_loss})
                losses['val'].append({'epoch': epoch, 'loss': val_loss})

                if self.cfg['output_configuration'].get('use_wandb'):
                    wandb_log({
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        **{f"train_{k}": v for k, v in train_metrics.items()},
                        **{f"val_{k}": v for k, v in val_metrics.items()}
                    })

                if epoch % self.cfg['output_configuration']['checkpoint_interval'] == 0:
                    log_and_checkpoint(self.model, self.optimizer, self.scheduler, epoch, val_metrics.get('base_accuracy', val_metrics.get('sean_accuracy', 0)), self.model_save_path, self.run_id)

                val_metric = val_metrics.get('base_accuracy', val_metrics.get('sean_accuracy', 0))
                if val_metric > self.best_val_metric:
                    log_and_checkpoint(self.model, self.optimizer, self.scheduler, epoch, val_metric, self.model_save_path, self.run_id, is_best=True)

                if self.check_early_stopping(val_metric):
                    break

        if GPUSetup.is_main_process():
            plot_loss_curves(losses, self.model_save_path, self.run_id)
            if latest_val_base_preds is not None and latest_val_base_labels is not None:
                if self.hierarchical_mode:
                    pred_labels = [f"{BASE_CLASSES[base]}{SUBCLASSES[subclass]}" if SUBCLASSES[subclass] else BASE_CLASSES[base]
                                   for base, subclass in zip(latest_val_base_preds.numpy(), latest_val_subclass_preds.numpy())]
                    true_labels = [f"{BASE_CLASSES[base]}{SUBCLASSES[subclass]}" if SUBCLASSES[subclass] else BASE_CLASSES[base]
                                   for base, subclass in zip(latest_val_base_labels.numpy(), latest_val_subclass_labels.numpy())]
                    plot_confusion_matrix(
                        {'true_labels': np.array(true_labels), 'pred_labels': np.array(pred_labels)},
                        self.model_save_path,
                        self.run_id
                    )
                else:
                    plot_confusion_matrix(
                        {'true_labels': latest_val_base_labels.numpy(), 'pred_labels': latest_val_base_preds.numpy()},
                        self.model_save_path,
                        self.run_id,
                        title_var="Sean_Review"
                    )
                    if not self.single_grader:
                        plot_confusion_matrix(
                            {'true_labels': latest_val_subclass_labels.numpy(), 'pred_labels': latest_val_subclass_preds.numpy()},
                            self.model_save_path,
                            self.run_id,
                            title_var="Santiago_Review"
                        )
            if self.cfg['output_configuration'].get('always_convert_best', False):
                final_checkpoint_conversion(self.model_save_path, self.run_id)

        if self.cfg['output_configuration'].get('use_wandb') and GPUSetup.is_main_process():
            import wandb
            wandb.finish()


class PretrainingEngine:
    def __init__(self, model, loader, cfg, run_path, device, pretrain_method, start_epoch=0):
        """
        Engine for pretraining methods like contrastive learning or masked autoencoders.

        Args:
            model: The model to train.
            loader: DataLoader for pretraining data.
            cfg: Configuration dictionary.
            run_path: Path for saving checkpoints and logs.
            device: Device to run training on.
            pretrain_method: Pretraining method ('contrastive' or 'mae').
            start_epoch: Starting epoch for resuming training.
        """
        self.model = model.to(device)
        self.loader = loader
        self.cfg = cfg
        self.run_path = run_path
        self.device = device
        self.pretrain_method = pretrain_method
        self.start_epoch = start_epoch
        self.use_amp = cfg['training']['module'].get('use_amp', False)
        self.scaler = GradScaler(enabled=self.use_amp)
        self.clip_grad = cfg['training']['module'].get('clip_grad', 1.0)
        self.grad_accum = cfg['training']['module'].get('grad_accum', 1)
        self.model_save_path, self.run_id = self.setup_experiment_environment()
                
        if pretrain_method == 'contrastive':
            self.criterion = SimCLRContrastiveLoss(
                temperature=cfg['training'].get('ssl_temperature', 0.07)
            )
        elif pretrain_method == 'moco':
            # pass the queue buffer from the model
            qbuf = model.module.queue if hasattr(model, 'module') else model.queue
            self.criterion = MoCoLoss(
                queue=qbuf,
                temperature=cfg['training'].get('ssl_temperature', 0.07)
            )
        elif pretrain_method == 'mae':
            self.criterion = ReconstructionLoss()
        else:
            raise ValueError(f"Unknown pretrain_method: {pretrain_method}")

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['ssl_lr'], weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg['training']['ssl_epochs'])

    def setup_experiment_environment(self):
        model_save_path = self.run_path
        run_id = datetime.now().strftime("%Y%m%d-%H%M")
        if self.cfg['output_configuration'].get('use_wandb') and GPUSetup.is_main_process():
            import wandb
            wandb.init(
                project=self.cfg['output_configuration']['task_name'],
                config=self.cfg['training'],
                name=run_id,
                tags=['pretrain', self.cfg['experiment']['name'], self.pretrain_method]
            )
        return model_save_path, run_id

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        batch_count = 0

        for batch_idx, batch in enumerate(self.loader):
            
            if self.pretrain_method == 'contrastive':
                v1 = batch['video1'].to(self.device)
                v2 = batch['video2'].to(self.device)
                with autocast(enabled=self.use_amp):
                    q, k = self.model(v1, v2)
                    loss = self.criterion(q, k)

            elif self.pretrain_method == 'moco':
                v = batch['video1'].to(self.device)
                with autocast(enabled=self.use_amp):
                    q, k = self.model(v)                     # updates the queue internally
                    loss = self.criterion(q, k)

            elif self.pretrain_method == 'mae':
                mvid = batch['masked_video'].to(self.device)
                orig = batch['original_video'].to(self.device)
                mask = batch['mask'].to(self.device)
                # indices of *which* patches were masked
                midxs = batch['mask_indices'].to(self.device)

                with autocast(enabled=self.use_amp):
                    # pass the mask_indices so the model only scatters those predictions
                    recon = self.model(mvid, mask_indices=midxs)
                    loss = self.criterion(recon, orig, mask)

            self.scaler.scale(loss / self.grad_accum).backward()

            # compute loss, backward, optimizer.step(), scaler.update(), zero_grad()
            if (batch_idx + 1) % self.grad_accum == 0:
                # after loss.backward() and every grad_accum steps:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.scaler.step(self.optimizer)   # 1) optimizer
                self.scaler.update()               # 2) update scale
                if self.scheduler:
                    self.scheduler.step()          # 3) update LR
                self.optimizer.zero_grad()         # 4) zero grads

            total_loss += loss.item() * self.grad_accum
            batch_count += 1

            if batch_idx % 10 == 0 and GPUSetup.is_main_process():
                log_info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        if GPUSetup.is_distributed():
            total_loss_tensor = torch.tensor(total_loss, device=self.device)
            torch.distributed.all_reduce(total_loss_tensor)
            total_loss = total_loss_tensor.item() / torch.distributed.get_world_size()
            batch_count_tensor = torch.tensor(batch_count, device=self.device)
            torch.distributed.all_reduce(batch_count_tensor)
            batch_count = batch_count_tensor.item() / torch.distributed.get_world_size()

        return total_loss / batch_count

    def train(self, epochs):
        losses = {'train': []}
        for epoch in range(self.start_epoch, epochs):
            if GPUSetup.is_distributed() and hasattr(self.loader.sampler, 'set_epoch'):
                self.loader.sampler.set_epoch(epoch)

            train_loss = self.train_epoch(epoch)

            if GPUSetup.is_main_process():
                log_info(f"Epoch {epoch}, Train Loss: {train_loss:.4f}")
                losses['train'].append({'epoch': epoch, 'loss': train_loss})

                if self.cfg['output_configuration'].get('use_wandb'):
                    wandb_log({
                        'train_loss': train_loss,
                    })

                if epoch % self.cfg['output_configuration']['checkpoint_interval'] == 0:
                    log_and_checkpoint(self.model, self.optimizer, self.scheduler, epoch, train_loss, self.model_save_path, self.run_id)

        if GPUSetup.is_main_process():
            plot_loss_curves(losses, self.model_save_path, self.run_id)
            if self.cfg['output_configuration'].get('always_convert_best', False):
                final_checkpoint_conversion(self.model_save_path, self.run_id)

        if self.cfg['output_configuration'].get('use_wandb') and GPUSetup.is_main_process():
            import wandb
            wandb.finish()

# -------------------------------------------------------------------------------- #

class EvalEngine:
    def __init__(self, model, eval_loader, cfg, run_path, device, save_attention=False):
        self.model = model.to(device)
        self.eval_loader = eval_loader
        self.cfg = cfg
        self.run_path = run_path
        self.device = device
        self.save_attention = save_attention and cfg['training']['model_name'] == 'vivit'
        self.use_amp = cfg['training']['module'].get('use_amp', False)
        self.model_save_path, self.run_id = self.setup_experiment_environment()
        self.hierarchical_mode = self._check_hierarchical_mode()
        self.single_grader = self._check_single_grader()

    def setup_experiment_environment(self):
        model_save_path = self.run_path
        run_id = datetime.now().strftime("%Y%m%d-%H%M")
        if self.cfg['output_configuration'].get('use_wandb') and GPUSetup.is_main_process():
            import wandb
            wandb.init(
                project=self.cfg['output_configuration']['task_name'],
                config=self.cfg['training'],
                name=run_id,
                tags=['eval', self.cfg['experiment']['name']]
            )
        return model_save_path, run_id

    def _check_hierarchical_mode(self):
        batch = next(iter(self.eval_loader))
        return isinstance(batch, dict) and 'base_label' in batch

    def _check_single_grader(self):
        batch = next(iter(self.eval_loader))
        if self.hierarchical_mode:
            return False
        return len(batch[1]) == 1 or (len(batch[1]) == 2 and torch.equal(batch[1][0], batch[1][1]))

    def process_batch(self, batch, batch_idx=0):
        if self.hierarchical_mode:
            videos = batch['video'].to(self.device)
            base_labels = batch['base_label'].to(self.device)
            subclass_labels = batch['subclass_label'].to(self.device)
        else:
            videos, labels = batch
            videos = videos.to(self.device)
            labels = [label.to(self.device) for label in labels]
            sean_labels = labels[0]
            santiago_labels = labels[0] if self.single_grader else labels[1]

        with torch.no_grad():
            with autocast(enabled=self.use_amp):
                outputs = self.model(videos)
                if self.hierarchical_mode:
                    base_preds = torch.argmax(outputs[0], dim=1)
                    subclass_preds = torch.argmax(outputs[1], dim=1)
                    preds = (base_preds, subclass_preds)
                else:
                    preds = torch.argmax(outputs, dim=1)

                attention_maps = None
                if self.save_attention:
                    attention_maps = self.model.get_attention_maps(videos)

        if self.hierarchical_mode:
            return outputs, preds, (base_labels, subclass_labels), attention_maps
        else:
            return outputs, preds, (sean_labels, santiago_labels), attention_maps

    def evaluate(self):
        self.model.eval()
        all_outputs = []
        all_preds = []
        all_labels = []

        for batch_idx, batch in enumerate(self.eval_loader):
            outputs, preds, labels, attention_maps = self.process_batch(batch, batch_idx)

            if self.hierarchical_mode:
                base_outputs, subclass_outputs = outputs
                base_preds, subclass_preds = preds
                base_labels, subclass_labels = labels
                all_outputs.append((base_outputs.detach().cpu(), subclass_outputs.detach().cpu()))
                all_preds.append((base_preds.detach().cpu(), subclass_preds.detach().cpu()))
                all_labels.append((base_labels.detach().cpu(), subclass_labels.detach().cpu()))
            else:
                all_outputs.append(outputs.detach().cpu())
                all_preds.append(preds.detach().cpu())
                all_labels.append((labels[0].detach().cpu(), labels[1].detach().cpu()))

            if self.save_attention and attention_maps is not None and GPUSetup.is_main_process():
                save_attention_maps(
                    attention_maps=attention_maps,
                    video_paths=[f"batch_{batch_idx}_sample_{i}" for i in range(len(batch[0]))],
                    save_dir=os.path.join(self.model_save_path, 'attention_maps'),
                    run_id=self.run_id,
                    batch_idx=batch_idx
                )

            if batch_idx % 10 == 0:
                log_info(f"Evaluation Batch {batch_idx}")

        if self.hierarchical_mode:
            all_base_outputs = torch.cat([x[0] for x in all_outputs])
            all_subclass_outputs = torch.cat([x[1] for x in all_outputs])
            all_outputs = (all_base_outputs, all_subclass_outputs)
            all_base_preds = torch.cat([x[0] for x in all_preds])
            all_subclass_preds = torch.cat([x[1] for x in all_preds])
            all_preds = (all_base_preds, all_subclass_preds)
            all_base_labels = torch.cat([x[0] for x in all_labels])
            all_subclass_labels = torch.cat([x[1] for x in all_labels])
            all_labels = (all_base_labels, all_subclass_labels)
        else:
            all_outputs = torch.cat(all_outputs)
            all_preds = torch.cat(all_preds)
            all_sean_labels = torch.cat([x[0] for x in all_labels])
            all_santiago_labels = torch.cat([x[1] for x in all_labels])
            all_labels = (all_sean_labels, all_santiago_labels)

        metrics = {}
        if self.hierarchical_mode:
            base_metrics = compute_metrics(all_base_labels, all_base_preds, self.cfg['training']['num_base_classes'])
            subclass_metrics = compute_metrics(all_subclass_labels, all_subclass_preds, self.cfg['training']['num_subclasses'])
            metrics.update({
                'base_accuracy': base_metrics['accuracy'],
                'base_f1': base_metrics['f1_per_class'],
                'base_precision': base_metrics['precision'],
                'base_recall': base_metrics['recall'],
                'subclass_accuracy': subclass_metrics['accuracy'],
                'subclass_f1': subclass_metrics['f1_per_class'],
                'subclass_precision': subclass_metrics['precision'],
                'subclass_recall': subclass_metrics['recall']
            })
        else:
            sean_metrics = compute_metrics(all_sean_labels, all_preds, self.cfg['training']['num_classes'])
            metrics.update({
                'sean_accuracy': sean_metrics['accuracy'],
                'sean_f1': sean_metrics['f1_per_class'],
                'sean_precision': sean_metrics['precision'],
                'sean_recall': sean_metrics['recall']
            })
            if not self.single_grader:
                santiago_metrics = compute_metrics(all_santiago_labels, all_preds, self.cfg['training']['num_classes'])
                metrics.update({
                    'santiago_accuracy': santiago_metrics['accuracy'],
                    'santiago_f1': santiago_metrics['f1_per_class'],
                    'santiago_precision': santiago_metrics['precision'],
                    'santiago_recall': santiago_metrics['recall']
                })

        if GPUSetup.is_distributed():
            metrics_tensor = torch.tensor([metrics[k] for k in sorted(metrics.keys()) if isinstance(metrics[k], (int, float))],
                                         dtype=torch.float32, device=self.device)
            torch.distributed.all_reduce(metrics_tensor, op=torch.distributed.ReduceOp.SUM)
            metrics_tensor /= GPUSetup.get_world_size()
            for idx, key in enumerate(sorted(metrics.keys())):
                if isinstance(metrics[key], (int, float)):
                    metrics[key] = metrics_tensor[idx].item()

        if GPUSetup.is_main_process():
            if self.cfg['output_configuration'].get('use_wandb'):
                wandb_log({f"eval_{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})
            self.post_evaluation_actions(metrics, all_labels, all_preds)

        return metrics

    def post_evaluation_actions(self, metrics, true_labels, pred_labels):
        if not GPUSetup.is_main_process():
            return
        if self.hierarchical_mode:
            all_base_labels, all_subclass_labels = true_labels
            all_base_preds, all_subclass_preds = pred_labels
            pred_labels = [f"{BASE_CLASSES[base.item()]}{SUBCLASSES[subclass.item()]}" if SUBCLASSES[subclass.item()] else BASE_CLASSES[base.item()]
                           for base, subclass in zip(all_base_preds, all_subclass_preds)]
            true_labels = [f"{BASE_CLASSES[base.item()]}{SUBCLASSES[subclass.item()]}" if SUBCLASSES[subclass.item()] else BASE_CLASSES[base.item()]
                           for base, subclass in zip(all_base_labels, all_subclass_labels)]
            plot_confusion_matrix(
                metrics={'true_labels': np.array(true_labels), 'pred_labels': np.array(pred_labels)},
                model_save_path=self.model_save_path,
                run_id=self.run_id
            )
        else:
            all_sean_labels, all_santiago_labels = true_labels
            plot_confusion_matrix(
                metrics={'true_labels': all_sean_labels.numpy(), 'pred_labels': pred_labels.numpy()},
                model_save_path=self.model_save_path,
                run_id=self.run_id,
                title="Confusion Matrix (Sean_Review)"
            )
            if not self.single_grader:
                plot_confusion_matrix(
                    metrics={'true_labels': all_santiago_labels.numpy(), 'pred_labels': pred_labels.numpy()},
                    model_save_path=self.model_save_path,
                    run_id=self.run_id,
                    title="Confusion Matrix (Santiago_Review)"
                )





            # elif self.pretrain_method == 'mae':
            #     mvid = batch['masked_video'].to(self.device)
            #     orig = batch['original_video'].to(self.device)
            #     mask = batch['mask'].to(self.device)
            #     with autocast(enabled=self.use_amp):
            #         recon = self.model(mvid)
            #         loss = self.criterion(recon, orig, mask)


                # self.scaler.unscale_(self.optimizer)
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                # self.scaler.step(self.optimizer)
                # self.scaler.update()
                # self.optimizer.zero_grad()
                # if self.scheduler:
                #     self.scheduler.step()

                    # self.scaler.step(self.optimizer)
                    # self.scaler.update()
                    # self.optimizer.zero_grad()
