import numpy as np
import pytorch_lightning as pl
from typing import Optional
import torch
import torch.nn as nn
import warnings
import transformers.optimization
from torch.optim import AdamW
import monai.networks.nets
import monai.losses
from ..metric import torch as _metric_torch_module


__all__ = ["CSFDModule", "CSFDSemanticSegmentationModule", "CSFDCroppedModule"]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CSFDModule(pl.LightningModule):
    def _get_model_from_cfg(self):
        if hasattr(monai.networks.nets, self.cfg.model.name):
            model = getattr(monai.networks.nets, self.cfg.model.name)
        else:
            available_model_names = [name for name in dir(monai.networks.nets) if not name.startswith("_")]
            raise NotImplementedError(f"""
{self.cfg.model.name} not implemented in monai.networks.nets
Available net_name: {available_model_names}
""")
        return model

    @property
    def num_classes(self):
        return len(self.cfg.dataset.target_columns)
    
    def _create_model(self, n_model_outputs) -> torch.nn.Module:
        model = self._get_model_from_cfg()
        return model(
            **self.cfg.model.kwargs,
            spatial_dims=self.cfg.model.spatial_dims,
            num_classes=n_model_outputs        
        )
    
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        if self.cfg.model.use_multi_sample_dropout:
            n_model_outputs = 100
            self.dropouts = [nn.Dropout(p) for p in np.linspace(0.1, 0.5, 5)]
            self.Linear = nn.Linear(n_model_outputs, self.num_classes)
        else:
            n_model_outputs = self.num_classes

        self.model = self._create_model(n_model_outputs)
        
        self.train_loss_meter = AverageMeter()

        self.loss_func = _metric_torch_module.competition_loss_with_logits

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            if self.cfg.train.evaluate_after_steps > 0:
                self.trainer.limit_val_batches = 0

            if self.cfg.model.name == "resnet10" and self.cfg.model.use_medical_net:
                print("[Info] Load model pretrained by MedicalNet")
                state_dict = self.model.state_dict()
                state_dict.update({
                    k.replace("module.", ""): v
                    for k, v in torch.load("../MedicalNet/pretrain/resnet_10_23dataset.pth")["state_dict"].items()
                })
                self.model.load_state_dict(state_dict)

            for n, param in self.named_parameters():
                param.requires_grad = True

    def forward(self, batch: dict) -> torch.Tensor:
        logits = self.model(batch["data"])
        if self.cfg.model.name == "ViT":
            logits = logits[0]

        if self.cfg.model.use_multi_sample_dropout:
            logits = torch.mean(
                torch.stack([
                    self.Linear(dropout(logits)) for dropout in self.dropouts
                ], dim=0),
                dim=0
            )
        return logits

    def _log_train_loss(self, loss, n_losses):
        if torch.isfinite(loss):
            self.train_loss_meter.update(loss.item(), n_losses)
            self.log("train/loss", self.train_loss_meter.avg, prog_bar=False)
        else:
            warnings.warn(f"""
        nan/inf detected: loss = {loss}
        """, UserWarning)

    def training_step(self, batch, batch_idx) -> dict:
        logits = self.forward(batch)
        loss = self.loss_func(logits, batch["label"])
        return {"loss": loss, "logits": logits}

    def training_step_end(self, batch_parts: dict):
        self._log_train_loss(batch_parts["loss"], len(batch_parts["logits"]))
        if (
            self.global_step >= self.cfg.train.evaluate_after_steps
            and self.trainer.limit_val_batches == 0.0
        ):
            self.trainer.limit_val_batches = 1.0

    def _log_validation_loss(self, loss):
        if torch.isfinite(loss):
            self.log("valid/loss", loss, prog_bar=False)
        else:
            warnings.warn(f"""
        nan/inf detected: loss = {loss}
        """, UserWarning)

    def validation_step(self, batch, batch_idx) -> dict:
        logits = self.forward(batch)
        loss = self.loss_func(logits, batch["label"])
        return {"loss": loss}
        
    def validation_step_end(self, batch_parts: dict):
        self._log_validation_loss(batch_parts["loss"])

    def configure_optimizers(self):
        if self.cfg.model.optimizer.name == "AdamW":
            optimizer = AdamW(
                self.parameters(),
                lr=self.cfg.train.learning_rate,
                weight_decay=self.cfg.train.weight_decay
            )
        else:
            raise NotImplementedError(self.cfg.model.optimizer.name)

        if self.cfg.model.optimizer.scheduler.name is None:
            return [optimizer]
        else:
            if self.cfg.model.optimizer.scheduler.name == "MultiStep":
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, **self.cfg.model.optimizer.scheduler.kwargs
                )
            elif self.cfg.model.optimizer.scheduler.name == "CosineAnnealingWarmRestarts":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, **self.cfg.model.optimizer.scheduler.kwargs
                )
            else:
                scheduler = transformers.get_scheduler(
                    self.cfg.model.optimizer.scheduler.name,
                    optimizer, **self.cfg.model.optimizer.scheduler.kwargs
                )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


class CSFDSemanticSegmentationModule(CSFDModule):
    def _create_model(self, n_model_outputs) -> torch.nn.Module:
        model = self._get_model_from_cfg()
        return model(
            **self.cfg.model.kwargs
        )

    def __init__(self, cfg):
        super().__init__(cfg)
        self.loss_func = monai.losses.DiceLoss(batch=True)
        self.metric_func = monai.metrics.DiceMetric()

    def training_step(self, batch, batch_idx) -> dict:
        logits = self.forward(batch)
        predicted = logits.sigmoid()
        true = batch["segmentation"]
        loss = self.loss_func(predicted, true)
        return {"loss": loss, "logits": logits}

    def validation_step(self, batch, batch_idx) -> dict:
        logits = self.forward(batch)

        true = torch.where(batch["segmentation"] > 0.5, 1, 0)
        predicted = torch.where(logits.sigmoid() > 0.5, 1, 0)

        metric = self.metric_func(predicted, true).mean(axis=0).mean(axis=0)
        loss = self.loss_func(logits.sigmoid(), true)
        return {"loss": loss, "metric": metric}
    
    def validation_step_end(self, batch_parts: dict):
        super(CSFDSemanticSegmentationModule, self).validation_step_end(batch_parts)
        self.log("valid/metric", batch_parts["metric"], prog_bar=False)


class CSFDCroppedModule(CSFDModule):
    @property
    def num_classes(self):
        return 1

    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.train.pos_weight is None:
            pos_weight = None
        else:
            pos_weight = torch.tensor(cfg.train.pos_weight)
        self.loss_func = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)

    def training_step(self, batch, batch_idx) -> dict:
        logits = self.forward(batch)
        loss = torch.mean(self.loss_func(logits, batch["label"]))
        return {"loss": loss, "logits": logits}

    def validation_step(self, batch, batch_idx) -> dict:
        logits = self.forward(batch)

        loss = torch.mean(self.loss_func(logits, batch["label"]))
        return {"loss": loss}
