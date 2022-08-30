from pytorch_lightning import Trainer, seed_everything
import pathlib
import torch
import re
import numpy as np
import CSFD.metric.numpy
import CSFD.data
import CSFD.data.three_dimensions
import gc
import tqdm


def load_cfg_and_checkpoints(model_path):
    model_path = pathlib.Path(model_path)
    _yaml_files = list(model_path.glob("*.yaml"))
    assert len(_yaml_files) == 1
    yaml_path = _yaml_files[0]

    cfg = CSFD.data.load_yaml_config(yaml_path)

    ckpt_dict = {}
    for ckpt_path in sorted(model_path.glob("checkpoints/*.ckpt")):
        matched = re.match(r".+fold(\d+)-of-\d+.+", ckpt_path.name)
        assert matched is not None
        fold = int(matched[1])
        ckpt_dict[fold] = ckpt_path

    return cfg, ckpt_dict


def validate_all_folds(cfg, ckpt_dict, df=None):
    if df is None:
        df = CSFD.data.three_dimensions.get_df(cfg.dataset)

    seed_everything(cfg.model.seed)

    cv_dict = {}
    for fold, checkpoint_path in ckpt_dict.items():
        print(f"* fold {fold}")
        cfg.dataset.cv.fold = fold
        cv_dict[cfg.dataset.cv.fold] = validate(cfg, checkpoint_path, df)

    print(np.mean(list(cv_dict.values())))
    return cv_dict


def validate(cfg, checkpoint_path, df):
    tl = Trainer(
        accelerator="gpu", devices=1,
        max_epochs=1000,
        precision=cfg.train.precision
    )
    module = CSFD.monai.CSFDModule.load_from_checkpoint(str(checkpoint_path), cfg=cfg)
    datamodule = CSFD.monai.CSFDDataModule(cfg, df)
    return tl.validate(module, datamodule)[0]["valid/loss"]


def predict_all_folds(cfg, ckpt_dict, df):
    seed_everything(cfg.model.seed)

    predicted_dict = {}
    for fold, ckpt_path in ckpt_dict.items():
        print(f"* fold {fold}")
        cfg.dataset.cv.fold = fold
        predicted_dict[fold] = predict(cfg, ckpt_path, df)
    return predicted_dict


def predict(cfg, ckpt_path, df):
    tl = Trainer(
        accelerator="gpu", devices=1,
        max_epochs=1000,
        precision=cfg.train.precision
    )
    module = CSFD.monai.CSFDModule.load_from_checkpoint(str(ckpt_path), cfg=cfg, map_location=torch.device("cuda"))
    datamodule = CSFD.monai.CSFDDataModule(cfg, df)
    predicted = _predict(tl, module, datamodule, use_sigmoid=True)
    del tl, module, datamodule
    gc.collect()
    torch.cuda.empty_cache()
    return predicted


def predict_on_datamodule_wide(cfg, ckpt_dict, df):
    modules = {
        fold: CSFD.monai.CSFDModule.load_from_checkpoint(
            str(ckpt_path), cfg=cfg, map_location=torch.device("cuda")
        ).cuda().half()
        for fold, ckpt_path in ckpt_dict.items()
    }
    datamodule = CSFD.monai.CSFDDataModule(cfg, df)

    datamodule.setup("predict")
    for batch in tqdm.tqdm(datamodule.predict_dataloader(), desc="predict"):
        predicted_list = []
        for fold, module in modules.items():
            with torch.no_grad():
                p = module.model.forward(
                    batch["data"].cuda()
                ).sigmoid()
            predicted_list.append(p.cpu().numpy())
        yield batch, predicted_list


def _predict(tl, module, datamodule, use_sigmoid):
    predicted = tl.predict(module, datamodule)
    with torch.no_grad():
        predicted = torch.concat(predicted).float()
        if use_sigmoid:
            predicted = predicted.sigmoid()
        predicted = predicted.numpy()
    return predicted
