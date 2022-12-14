import warnings

import omegaconf

import CSFD.monai
import CSFD.data
import CSFD.data.io.three_dimensions
import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import pathlib
from collections.abc import Iterable


def train(cfg, module_class, datamodule_class, df=None):
    if cfg.dataset.cv.fold is None:
        cfg.dataset.cv.fold = list(range(cfg.dataset.cv.n_folds))
        train(cfg, module_class, datamodule_class, df)
        return
    elif isinstance(cfg.dataset.cv.fold, Iterable):
        for fold in cfg.dataset.cv.fold:
            print(f"* fold {fold}")
            cfg.dataset.cv.fold = fold
            train(cfg, module_class, datamodule_class, df)
        return

    if df is None:
        df = CSFD.data.io_with_cfg.three_dimensions.get_df(cfg.dataset)
    seed_everything(cfg.train.seed)

    module = module_class(cfg)
    datamodule = datamodule_class(cfg, df)

    model_path = pathlib.Path(cfg.train.model_path)
    model_path.mkdir(exist_ok=True, parents=True)
    copied_yaml_path = model_path / f"{cfg.model.name}.yaml"
    if copied_yaml_path.exists():
        old_cfg = CSFD.data.io.load_yaml_config(copied_yaml_path)
        old_cfg.dataset.cv.fold = None
        copied_cfg = cfg.copy()
        copied_cfg.dataset.cv.fold = None
        if old_cfg != copied_cfg:
            warnings.warn(f"""
            old cfg: {old_cfg}
            cfg: {copied_cfg}
            """, UserWarning)
            raise FileExistsError(copied_yaml_path)
    else:
        omegaconf.OmegaConf.save(cfg, copied_yaml_path)
    filename = "_".join(
        [
            f"{cfg.train.name_prefix}{cfg.model.name}",
            f"fold{cfg.dataset.cv.fold}-of-{cfg.dataset.cv.n_folds}",
            *(
                [cfg.train.name_suffix] if cfg.train.name_suffix else []
            )
        ]
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=model_path / "checkpoints",
            filename=filename,
            verbose=True,
            monitor="valid/loss" if cfg.train.save_on_train_epoch_end is False else None,
            mode="min",
            save_weights_only=True,
            save_on_train_epoch_end=cfg.train.save_on_train_epoch_end
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
            name=filename
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
