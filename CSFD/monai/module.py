import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import transformers.optimization
from torchmetrics import F1Score
import monai.networks.nets


__all__ = ["Module"]


class Module(pl.LightningModule):
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
        self.model = model(
            pretrained=self.cfg.model.pretrained,
            spatial_dims=3, n_input_channels=1,
            num_classes=self.cfg.model.num_labels
        )
        self.name = self.model.__class__.__name__
        for n, param in self.model.named_parameters():
            param.requires_grad = True

        self.train_f1_score = F1Score(num_classes=self.cfg.model.num_labels)
        self.val_f1_score = F1Score(num_classes=self.cfg.model.num_labels)
        self.compute_loss = F.binary_cross_entropy_with_logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))

    # @staticmethod
    # def compute_loss(y_hat: Tensor, y: Tensor):
    #     return F.binary_cross_entropy_with_logits(y_hat, y.to(y_hat.dtype))

    def training_step(self, batch, batch_idx):
        img, y = batch["data"], batch["label"]
        y_hat = self.forward(img)
        loss = self.compute_loss(y_hat, y)
        self.log("train/loss", loss, prog_bar=False)
        self.log("train/f1", self.train_f1_score(y_hat, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, y = batch["data"], batch["label"]
        y_hat = self.forward(img)
        loss = self.compute_loss(y_hat, y)
        self.log("valid/loss", loss, prog_bar=False)
        self.log("valid/f1", self.val_f1_score(y_hat, y), prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.model.parameters(), lr=self.cfg.train.learning_rate)
        scheduler = transformers.optimization.get_scheduler(
            self.cfg.optimizer.scheduler.name, optimizer,
            num_warmup_steps=self.cfg.optimizer.scheduler.num_warmup_steps,
            num_training_steps=self.cfg.optimizer.scheduler.num_training_steps
        )
        return [optimizer], [scheduler]
