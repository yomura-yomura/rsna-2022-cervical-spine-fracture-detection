import numpy as np
import pytorch_lightning as pl
from typing import Optional
import torch
import torch.nn as nn
import warnings
import transformers.optimization
from transformers import AdamW
from torchmetrics import F1Score
import monai.networks.nets
from ..metric import torch as _metric_torch_module


__all__ = ["CSFDModule"]


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
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        if hasattr(monai.networks.nets, self.cfg.model.name):
            model = getattr(monai.networks.nets, self.cfg.model.name)
        else:
            available_model_names = [name for name in dir(monai.networks.nets) if not name.startswith("_")]
            raise NotImplementedError(f"""
{self.cfg.model.name} not implemented in monai.networks.nets
Available net_name: {available_model_names}
""")
        self.num_classes = len(self.cfg.dataset.target_columns)
        if self.cfg.model.use_multi_sample_dropout:
            n_model_outputs = 100
            self.dropouts = [nn.Dropout(p) for p in np.linspace(0.1, 0.5, 5)]
            self.Linear = nn.Linear(n_model_outputs, self.num_classes)
        else:
            n_model_outputs = self.num_classes

        # assert self.cfg.model.pretrained is False
        self.model: torch.nn.Module = model(
            **self.cfg.model.kwargs,
            spatial_dims=3,
            num_classes=n_model_outputs
        )

        self.train_f1_score = F1Score(num_classes=self.num_classes)
        self.val_f1_score = F1Score(num_classes=self.num_classes)

        self.train_loss_meter = AverageMeter()

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            if self.cfg.train.evaluate_after_steps > 0:
                self.trainer.limit_val_batches = 0

            if self.cfg.model.name == "resnet10":
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

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch)
        loss = _metric_torch_module.competition_loss_with_logits(logits, batch["label"])

        if torch.isfinite(loss):
            self.train_loss_meter.update(loss.item(), len(logits))
            self.log("train/loss", self.train_loss_meter.avg, prog_bar=False)
        else:
            warnings.warn(f"""
nan/inf detected: loss = {loss}
logits = {logits}
label = {batch["label"]}
            """, UserWarning)
            self.log("train/loss", loss.item(), prog_bar=False)
        return loss

    def training_step_end(self, _):
        if (
            self.global_step >= self.cfg.train.evaluate_after_steps
            and self.trainer.limit_val_batches == 0.0
        ):
            self.trainer.limit_val_batches = 1.0

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch)
        loss = _metric_torch_module.competition_loss_with_logits(logits, batch["label"])
        if not torch.isfinite(loss):
            warnings.warn(f"""
nan/inf detected: loss = {loss}
logits = {logits}
label = {batch["label"]}
            """)
        return loss

    def validation_step_end(self, loss):
        self.log("valid/loss", float(torch.nanmean(loss)), prog_bar=False)

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
