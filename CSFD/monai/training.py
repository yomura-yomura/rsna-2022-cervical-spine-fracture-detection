import CSFD.monai
import CSFD.data
import CSFD.data.three_dimensions
import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import pathlib


def train(cfg):
    if cfg.dataset.cv.fold is None:
        for fold in range(cfg.dataset.cv.n_folds):
            print(f"* fold {fold}")
            cfg.dataset.cv.fold = fold
            train(cfg)
        return

    df = CSFD.data.three_dimensions.get_df(cfg.dataset)
    seed_everything(cfg.train.seed)

    module = CSFD.monai.CSFDModule(cfg)
    datamodule = CSFD.monai.CSFDDataModule(cfg, df)

    name = f"{cfg.train.name_prefix}{cfg.model.name}_fold{cfg.dataset.cv.fold}-of-{cfg.dataset.cv.n_folds}_{cfg.train.name_suffix}"

    callbacks = [
        ModelCheckpoint(
            dirpath=pathlib.Path(cfg.train.model_path) / "checkpoints",
            filename=name,
            monitor="valid/loss",
            mode="min",
            save_weights_only=True,
            # save_last=not cfg.train.save_best_checkpoint,
        ),
        LearningRateMonitor("step"),
    ]
    if cfg.train.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor='valid/loss',
                min_delta=0.00,
                patience=3,
                verbose=False,
                mode='min'
            )
        )

    wandb.login()

    tl = Trainer(
        logger=WandbLogger(
            project="Cervical-Spine-Fracture-Detection",
            name=name
            # config=cfg
        ),
        callbacks=callbacks,
        # max_steps=config.optim.scheduler.num_training_steps,
        max_epochs=cfg.train.max_epochs,
        # amp_backend=amp_backend,
        # max_steps=cfg.model.optimizer.scheduler.num_training_steps,

        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision,
        # gradient_clip_val=config.train.max_grad_norm,
        val_check_interval=min(cfg.train.validation_interval, 1.0),
        check_val_every_n_epoch=max(int(cfg.train.validation_interval), 1),
        # limit_val_batches=0.0 if cfg.train.evaluate_after_steps > 0 else 1.0,
        # accumulate_grad_batches=cfg.train.accumulate_grads,
        log_every_n_steps=cfg.train.logging_interval,
    )
    tl.fit(module, datamodule)

    wandb.finish()
