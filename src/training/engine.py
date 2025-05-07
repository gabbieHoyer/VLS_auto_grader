# src/training/engine.py
import logging
import numpy as np
from datetime import datetime

import torch
from torch.cuda.amp import autocast, GradScaler

from src.utils import (
    GPUSetup, log_info, wandb_log, 
    log_and_checkpoint, final_checkpoint_conversion,
    plot_confusion_matrix, plot_loss_curves, save_attention_maps,
    plot_augmentations, plot_pretrain_augmentations, plot_qc_predictions,
    init_wandb_run,
)
from src.models import compute_metrics, flatten_metrics
from src.ssl import SimCLRContrastiveLoss, MoCoLoss, ReconstructionLoss

logger = logging.getLogger(__name__)

# Mapping for reconstructing labels in hierarchical case
BASE_CLASSES = {0: '1', 1: '2', 2: '3', 3: '4'}
SUBCLASSES = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: ''}


class TrainingEngine:
    def __init__(self, model, optimizer, scheduler, criterion, train_loader, val_loader, cfg, run_id, run_path, device, start_epoch=0):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        # self.run_path = run_path
        self.device = device
        self.start_epoch = start_epoch
        self.use_amp = cfg['training']['module'].get('use_amp', False)
        self.scaler = GradScaler(enabled=self.use_amp)
        self.clip_grad = cfg['training']['module'].get('clip_grad', 1.0)
        self.grad_accum = cfg['training']['module'].get('grad_accum', 1)
        self.early_stopping = cfg['training']['module'].get('early_stopping', {})
        self.best_val_metric = 0.0
        self.no_improve_epochs = 0

        self.hierarchical_mode = self._check_hierarchical_mode()
        self.single_grader = self._check_single_grader()

        ds = self.train_loader.dataset
        self.idx_to_class    = getattr(ds, 'idx_to_class', None)
        self.idx_to_base     = getattr(ds, 'idx_to_base', None)
        self.idx_to_subclass = getattr(ds, 'idx_to_subclass', None)

        self.run_id = run_id
        self.model_save_path = run_path

        # only on the main process, kick off wandb
        if cfg['output_configuration'].get('use_wandb') and GPUSetup.is_main_process():
            # for TrainingEngine it’s always finetune
            self.wandb_run = init_wandb_run(
                cfg, self.run_id, 
                stage='finetune',
                tags=['train', cfg['experiment']['name'], 'supervised']
            )

    def _flatten_with_names(self, prefix, metrics, out_dict):
            """
            Like flatten_metrics, but replaces 'class_{i}' with real label names.
            """
            for name, value in metrics.items():
                # pull numpy array or scalar
                if isinstance(value, torch.Tensor):
                    arr = value.detach().cpu().numpy()
                else:
                    arr = value

                # array-like → unroll per index
                if hasattr(arr, "__len__") and not isinstance(arr, (str, bytes)):
                    for i, v in enumerate(arr):
                        # pick the right map
                        if name in ("sean_f1", "santiago_f1"):
                            label = self.idx_to_class.get(i, f"class_{i}")
                        elif name == "base_f1":
                            label = self.idx_to_base.get(i, f"{i+1}")
                        elif name == "subclass_f1":
                            label = self.idx_to_subclass.get(i, f"sub{i}")
                        else:
                            label = f"class_{i}"
                        out_dict[f"{prefix}_{name}_{label}"] = float(v)
                else:
                    # scalar
                    out_dict[f"{prefix}_{name}"] = float(arr)

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

                # for the very first batch, do QC *before* any autocast/grad activity
                if batch_idx == 0 and epoch == self.start_epoch:
                    with torch.no_grad():
                        plot_qc_predictions(
                            loader     = self.val_loader,
                            model      = self.model,
                            device     = self.device,
                            run_path   = self.model_save_path,
                            fig_dir    = self.cfg['paths']['figures_dir'],
                            run_id     = self.run_id,
                            prefix     = 'val_qc',
                            n_samples  = 4,
                            n_frames   = 8
                      )

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

            total_loss += loss.item() * self.grad_accum

            batch_count += 1
            if batch_idx % 10 == 0 and GPUSetup.is_main_process():
                log_info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        #  Flush any leftover gradients at end of epoch
        if (batch_idx + 1) % self.grad_accum != 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

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

    def validate(self, epoch):
        if self.val_loader is None:
            return 0, {}, None, None, None, None
        
        if GPUSetup.is_main_process() and epoch == self.start_epoch:
            plot_augmentations(
                loader=self.val_loader,
                run_path=self.model_save_path,
                fig_dir=self.cfg['paths']['figures_dir'],
                run_id=self.run_id,
                prefix='val',
                n_frames=8,
                save_path=self.model_save_path
            )

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
            
            # still should have sean vs santiago splits in hierarchical mode = true******************
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

        if GPUSetup.is_main_process():
            plot_augmentations(
                loader=self.train_loader,
                run_path=self.model_save_path,
                fig_dir=self.cfg['paths']['figures_dir'],
                run_id=self.run_id,
                prefix='train',
                n_frames=8,
                save_path=self.model_save_path
            )

        for epoch in range(self.start_epoch, epochs):
            if GPUSetup.is_distributed() and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            train_loss, train_metrics = self.train_epoch(epoch)
            val_loss, val_metrics, val_base_preds, val_base_labels, val_subclass_preds, val_subclass_labels = self.validate(epoch)

            if val_base_preds is not None and val_base_labels is not None:
                latest_val_base_preds, latest_val_base_labels = val_base_preds, val_base_labels
                latest_val_subclass_preds, latest_val_subclass_labels = val_subclass_preds, val_subclass_labels

            # ---- step the scheduler based on its type ----
            if self.scheduler is not None:
                # get the scheduler class name
                sched_name = self.scheduler.__class__.__name__
                if sched_name == 'ReduceLROnPlateau':
                    # pass in the metric you’re watching
                    self.scheduler.step(val_loss)
                else:
                    # e.g. CosineAnnealingLR or StepLR
                    self.scheduler.step()

            if GPUSetup.is_main_process():
                log_info(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                log_info(f"Train Metrics: {train_metrics}")
                log_info(f"Val Metrics: {val_metrics}")
                losses['train'].append({'epoch': epoch, 'loss': train_loss})
                losses['val'].append({'epoch': epoch, 'loss': val_loss})

                if self.cfg['output_configuration'].get('use_wandb'):
                    log_dict = {
                            'train_loss': train_loss,
                            'val_loss':   val_loss,
                            'lr':         self.optimizer.param_groups[0]['lr']
                    }
                    # this will inject keys like train_sean_f1_2b etc.
                    self._flatten_with_names('train', train_metrics, log_dict)
                    self._flatten_with_names('val',   val_metrics,   log_dict)

                    wandb_log(log_dict, step=epoch)

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
    def __init__(self, model, pretrain_loader, cfg, run_id, run_path, device, pretrain_method, start_epoch=0):
        """
        Engine for pretraining methods like contrastive learning or masked autoencoders.

        Args:
            model: The model to train.
            pretrain_loader: DataLoader for pretraining data. (may match train set in some cases with limited data or fully labeled data)
            cfg: Configuration dictionary.
            run_path: Path for saving checkpoints and logs.
            device: Device to run training on.
            pretrain_method: Pretraining method ('contrastive' or 'mae').
            start_epoch: Starting epoch for resuming training.
        """
        self.model = model.to(device)
        self.pretrain_loader = pretrain_loader
        self.dataset = pretrain_loader.dataset  # Store reference to dataset
        self.cfg = cfg
        self.device = device
        self.pretrain_method = pretrain_method
        self.start_epoch = start_epoch
        self.use_amp = cfg['training']['module'].get('use_amp', False)
        self.scaler = GradScaler(enabled=self.use_amp)
        self.clip_grad = cfg['training']['module'].get('clip_grad', 1.0)
        self.grad_accum = cfg['training']['module'].get('grad_accum', 1)
 
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

        self.run_id = run_id
        self.model_save_path = run_path

        # only on the main process, kick off wandb
        if cfg['output_configuration'].get('use_wandb') and GPUSetup.is_main_process():
            # for TrainingEngine it’s always finetune
            self.wandb_run = init_wandb_run(
                cfg, self.run_id, 
                stage='pretrain',
                tags=['pretrain', cfg['experiment']['name'], self.pretrain_method]
                )
            
    def train_epoch(self, epoch):
        self.model.train()

        # 1) reset peak‐mem stats for this epoch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        total_loss = 0
        batch_count = 0

        for batch_idx, batch in enumerate(self.pretrain_loader):
            
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
                # 2) unscale, clip, step, LR, zero_grad
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.clip_grad
                )
                self.scaler.step(self.optimizer)   # 1) optimizer
                self.scaler.update()               # 2) update scale
                # if self.scheduler:
                #     self.scheduler.step()          # 3) update LR
                self.optimizer.zero_grad()         # 4) zero grads

                # 3) log peak memory after each step
                if torch.cuda.is_available() and GPUSetup.is_main_process():
                    peak = torch.cuda.max_memory_allocated(self.device) / (1024**2)
                    logger.info(f"[Epoch {epoch} Batch {batch_idx}] peak GPU mem: {peak:.1f} MB")

            total_loss += loss.item() * self.grad_accum
            batch_count += 1

            if batch_idx % 10 == 0 and GPUSetup.is_main_process():
                log_info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # 4) flush any leftover partial‐accumulated gradients
        if (batch_idx + 1) % self.grad_accum != 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            # log peak‐memory for the final partial step
            if torch.cuda.is_available() and GPUSetup.is_main_process():
                peak = torch.cuda.max_memory_allocated(self.device) / (1024**2)
                logger.info(f"[Epoch {epoch} Batch {batch_idx} (final flush)] peak GPU mem: {peak:.1f} MB")

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

        if GPUSetup.is_main_process():
            plot_pretrain_augmentations(
                loader=self.pretrain_loader,
                run_path=self.model_save_path,
                fig_dir=self.cfg['paths']['figures_dir'],
                run_id=self.run_id,
                pretrain_method=self.pretrain_method,
                n_frames=8,
                save_path=self.model_save_path
            )

        for epoch in range(self.start_epoch, epochs):
            # Set the epoch on the dataset for curriculum learning (e.g., MAE mask ratio)
            # Set the epoch on the dataset for all pretraining methods
            self.dataset.set_epoch(epoch)

            # Update mask_ratio for MAE
            current_mask_ratio = None  # For WandB logging
            if self.pretrain_method == 'mae':
                mask_ratio = self.dataset.mask_ratio
                end_mask_ratio = self.dataset.end_mask_ratio
                total_epochs = self.dataset.total_epochs

                # Validate total_epochs to prevent division by zero
                if total_epochs <= 1:
                    raise ValueError(f"total_epochs must be > 1, got {total_epochs}")

                progress = epoch / (total_epochs - 1)
                current_mask_ratio = mask_ratio + (end_mask_ratio - mask_ratio) * progress

                # Clamp mask_ratio to [0, 1]
                current_mask_ratio = max(0.0, min(1.0, current_mask_ratio))

                model = self.model.module if hasattr(self.model, 'module') else self.model
                model.set_mask_ratio(current_mask_ratio)
                
                if GPUSetup.is_main_process():
                    logger.info(f"Epoch {epoch}: mask_ratio={current_mask_ratio:.3f}")
                
            # Set epoch for distributed sampler
            if GPUSetup.is_distributed() and hasattr(self.pretrain_loader.sampler, 'set_epoch'):
                self.pretrain_loader.sampler.set_epoch(epoch)

            train_loss = self.train_epoch(epoch)

            # one scheduler step per epoch
            if self.scheduler:
                self.scheduler.step()

            if GPUSetup.is_main_process():
                log_info(f"Epoch {epoch}, Train Loss: {train_loss:.4f}")
                losses['train'].append({'epoch': epoch, 'loss': train_loss})

                if self.cfg['output_configuration'].get('use_wandb'):
                    wandb_log({
                        'train_loss': train_loss,
                        'mask_ratio': current_mask_ratio if self.pretrain_method == 'mae' else None,
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
# -------------------------------------------------------------------------------- #
class EvalEngine:
    def __init__(
        self, model, test_loader, cfg, device,
        save_attention=False, output_dir=None
    ):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.cfg = cfg
        self.device = device
        self.save_attention = save_attention
        self.output_dir = output_dir
        self.use_amp = cfg['training']['module'].get('use_amp', False)

        # detect modes
        self.hierarchical = isinstance(
            next(iter(test_loader)), dict
        ) and 'base_label' in next(iter(test_loader))
        self.single_grader = False
        if not self.hierarchical:
            batch = next(iter(test_loader))
            self.single_grader = (
                len(batch[1]) == 1 or 
                (len(batch[1]) == 2 and torch.equal(batch[1][0], batch[1][1]))
            )

        # wandb
        self.use_wandb = cfg['output_configuration'].get('use_wandb', False)
        if self.use_wandb and GPUSetup.is_main_process():
            from utils.wandb_utils import init_wandb_run

            # have option where the run_id is passed direclty from previous training vs if this was separate time completely and need to load in that info once more
            # metadata = load_run_metadata(run_folder)
            # self.wandb = init_wandb_run(
            #     cfg, metadata['run_id'],
            #     stage='eval',
            #     tags=['eval', cfg['experiment']['name']]
            # )

    def evaluate(self):
        self.model.eval()
        all_base_preds, all_sub_preds = [], []
        all_base_labels, all_sub_labels = [], []

        with torch.no_grad():
            for idx, batch in enumerate(self.test_loader):
                if self.hierarchical:
                    videos = batch['video'].to(self.device)
                    base_labels = batch['base_label'].to(self.device)
                    sub_labels = batch['subclass_label'].to(self.device)
                    with autocast(enabled=self.use_amp):
                        out_base, out_sub = self.model(videos)
                    preds_base = torch.argmax(out_base, dim=1)
                    preds_sub = torch.argmax(out_sub, dim=1)
                else:
                    videos, labels = batch
                    videos = videos.to(self.device)
                    labels = [l.to(self.device) for l in labels]
                    base_labels = labels[0]
                    with autocast(enabled=self.use_amp):
                        outputs = self.model(videos)
                    preds_base = torch.argmax(outputs, dim=1)
                    if not self.single_grader:
                        preds_sub = preds_base.clone()
                        sub_labels = labels[1]
                    else:
                        preds_sub = preds_base.clone()
                        sub_labels = base_labels

                all_base_preds.append(preds_base.cpu())
                all_base_labels.append(base_labels.cpu())
                all_sub_preds.append(preds_sub.cpu())
                all_sub_labels.append(sub_labels.cpu())

                if self.save_attention:
                    save_attention_maps(
                        inputs=videos, outputs=
                        (out_base, out_sub) if self.hierarchical else outputs,
                        save_dir=self.output_dir, batch_idx=idx
                    )

        # flatten
        base_p = torch.cat(all_base_preds)
        base_l = torch.cat(all_base_labels)
        sub_p = torch.cat(all_sub_preds)
        sub_l = torch.cat(all_sub_labels)

        # compute
        metrics = {}
        if self.hierarchical:
            base_m = compute_metrics(base_l.numpy(), base_p.numpy(), self.cfg['training']['num_base_classes'])
            sub_m = compute_metrics(sub_l.numpy(), sub_p.numpy(), self.cfg['training']['num_subclasses'])
            metrics = {
                'base_accuracy': base_m['accuracy'],
                'base_f1': base_m['f1_per_class'],
                'sub_accuracy': sub_m['accuracy'],
                'sub_f1': sub_m['f1_per_class']
            }
        else:
            base_m = compute_metrics(base_l.numpy(), base_p.numpy(), self.cfg['training']['num_classes'])
            metrics = {
                'base_accuracy': base_m['accuracy'],
                'base_f1': base_m['f1_per_class']
            }
            if not self.single_grader:
                sub_m = compute_metrics(sub_l.numpy(), sub_p.numpy(), self.cfg['training']['num_classes'])
                metrics.update({
                    'sub_accuracy': sub_m['accuracy'],
                    'sub_f1': sub_m['f1_per_class']
                })

        # log
        if self.use_wandb and GPUSetup.is_main_process():
            wandb_log(metrics)

        return metrics


# -------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------- #
