import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import torch.nn as nn
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
        if not hasattr(cfg.model, "use_multi_sample_dropout"):
            cfg.model.use_multi_sample_dropout = False

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
            n_model_outputs = 1000
            self.dropouts = [nn.Dropout(p) for p in np.linspace(0.1, 0.5, 5)]
            self.Linear = nn.Linear(n_model_outputs, self.num_classes)
        else:
            n_model_outputs = self.num_classes

        # assert self.cfg.model.pretrained is False
        self.model = model(
            **self.cfg.model.kwargs,
            spatial_dims=3,
            num_classes=n_model_outputs
        )

        # self.name = self.model.__class__.__name__
        for n, param in self.model.named_parameters():
            param.requires_grad = True

        self.train_f1_score = F1Score(num_classes=self.num_classes)
        self.val_f1_score = F1Score(num_classes=self.num_classes)

        self.train_loss_meter = AverageMeter()

    def forward(self, batch: dict) -> torch.Tensor:
        logits = self.model(batch["data"])
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
        loss = _metric_torch_module.competition_loss(logits, batch["label"])

        self.train_loss_meter.update(loss.item(), self.num_classes)
        self.log("train/loss", self.train_loss_meter.avg, prog_bar=False)
        # self.log("train/f1", self.train_f1_score(y_hat, batch["label"].to(int)), prog_bar=True)
        return loss

    def training_step_end(self, outputs):
        if (
            self.global_step >= self.cfg.train.evaluate_after_steps
            and self.trainer.limit_val_batches == 0.0
        ):
            self.trainer.limit_val_batches = 1.0

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch)
        loss = _metric_torch_module.competition_loss(logits, batch["label"])
        self.log("valid/loss", loss, prog_bar=False)
        # self.log("valid/f1", self.val_f1_score(y_hat, batch["label"].to(int)), prog_bar=True)

    def configure_optimizers(self):
        if self.cfg.model.optimizer.name == "AdamW":
            optimizer = AdamW(
                # self.parameters(),
                self.model.parameters(),
                lr=self.cfg.train.learning_rate,
                weight_decay=self.cfg.train.weight_decay
            )
        else:
            raise NotImplementedError(self.cfg.model.optimizer.name)

        if self.cfg.model.optimizer.scheduler.name is None:
            return [optimizer]
        else:
            scheduler = transformers.get_scheduler(
                self.cfg.model.optimizer.scheduler.name,
                optimizer,
                num_warmup_steps=self.cfg.model.optimizer.scheduler.num_warmup_steps,
                num_training_steps=self.cfg.model.optimizer.scheduler.num_training_steps
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
