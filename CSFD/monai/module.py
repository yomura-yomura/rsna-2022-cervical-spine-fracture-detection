from typing import Any, Optional, Sequence, Tuple, Type, Union
import pytorch_lightning as pl
from torch import nn, Tensor
import torch.nn.functional as F
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
from torchmetrics import F1Score
import monai.networks.nets


class LitNeckCT(pl.LightningModule):
    def __init__(
        self, num_labels: int = 7, lr: float = 1e-3, optimizer: Optional[Type[Optimizer]] = None,
        net_name="resnet10"
    ):
        super().__init__()
        self.net = getattr(monai.networks.nets, net_name)(
            pretrained=False, spatial_dims=3, n_input_channels=1, num_classes=num_labels
        )
        self.name = self.net.__class__.__name__
        for n, param in self.net.named_parameters():
            param.requires_grad = True
        self.learning_rate = lr
        self.optimizer = optimizer or AdamW

        self.train_f1_score = F1Score(num_classes=num_labels)
        self.val_f1_score = F1Score(num_classes=num_labels)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.net(x))

    @staticmethod
    def compute_loss(y_hat: Tensor, y: Tensor):
        return F.binary_cross_entropy_with_logits(y_hat, y.to(y_hat.dtype))

    def training_step(self, batch, batch_idx):
        img, y = batch["data"], batch["label"]
        y_hat = self(img)
        loss = self.compute_loss(y_hat, y)
        self.log("train/loss", loss, prog_bar=False)
        self.log("train/f1", self.train_f1_score(y_hat, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, y = batch["data"], batch["label"]
        y_hat = self(img)
        loss = self.compute_loss(y_hat, y)
        self.log("valid/loss", loss, prog_bar=False)
        self.log("valid/f1", self.val_f1_score(y_hat, y), prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.net.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, self.trainer.max_epochs * 200, 0)
        return [optimizer], [scheduler]
