from pytorch_lightning import Trainer, seed_everything
import pathlib
import torch
import re
import numpy as np
import CSFD.metric.numpy
import CSFD.data
import CSFD.data.io.three_dimensions
import gc
import tqdm


def load_cfg_and_checkpoints(model_path):
    model_path = pathlib.Path(model_path)
    _yaml_files = list(model_path.glob("*.yaml"))
    assert len(_yaml_files) == 1
    yaml_path = _yaml_files[0]

    cfg = CSFD.data.io.load_yaml_config(yaml_path)

    ckpt_dict = {}
    for ckpt_path in sorted(model_path.glob("checkpoints/*.ckpt")):
        matched = re.match(r".+fold(\d+)-of-\d+.+", ckpt_path.name)
        assert matched is not None
        fold = int(matched[1])
        ckpt_dict[fold] = ckpt_path

    return cfg, ckpt_dict


def validate_all_folds(cfg, ckpt_dict, df=None):
    if df is None:
        df = CSFD.data.io.three_dimensions.get_df(cfg.dataset)

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


# def predict_all_folds(cfg, ckpt_dict, df):
#     seed_everything(cfg.model.seed)
#
#     predicted_dict = {}
#     for fold, ckpt_path in ckpt_dict.items():
#         print(f"* fold {fold}")
#         cfg.dataset.cv.fold = fold
#         predicted_dict[fold] = predict(cfg, ckpt_path, df)
#     return predicted_dict


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


def predict_on_datamodule_wide(cfg, ckpt_dict, df, common_cfg_dataset, show_progress=True):
    if isinstance(cfg, list) or isinstance(ckpt_dict, list):
        assert isinstance(cfg, list) and isinstance(ckpt_dict, list)
        assert len(cfg) == len(ckpt_dict)

        ckpt_dict_list = ckpt_dict
        ckpt_dict = {
            fold: [ckpt_dict[fold] for ckpt_dict in ckpt_dict_list]
            for fold in set.intersection(*(set(ckpt_dict.keys()) for ckpt_dict in ckpt_dict_list))
        }
    else:
        ckpt_dict = {
            fold: [ckpt_path]
            for fold, ckpt_path in ckpt_dict.items()
        }

    cfg_list = [cfg] if not isinstance(cfg, list) else cfg

    modules_dict = {
        fold: [
            CSFD.monai.CSFDModule.load_from_checkpoint(
                str(ckpt_path), cfg=cfg, map_location=torch.device("cuda")
            ).cuda().half()
            for ckpt_path, cfg in zip(ckpt_path_list, cfg_list)
        ]
        for fold, ckpt_path_list in ckpt_dict.items()
    }
    for modules in modules_dict.values():
        for module in modules:
            module.eval()

    datamodule = CSFD.monai.CSFDDataModule(common_cfg_dataset, df)

    datamodule.setup("predict")
    dataloader = (
        tqdm.tqdm(datamodule.predict_dataloader(), desc="predict")
        if show_progress else
        datamodule.predict_dataloader()
    )

    for batch in dataloader:
        predicted_list = []
        for fold, modules in modules_dict.items():
            predicted_per_module_list = []
            for module in modules:
                with torch.no_grad():
                    p = module.forward(
                        {"data": batch["data"].cuda()}
                    ).sigmoid()
                    predicted_per_module_list.append(p.cpu().numpy())
            predicted_list.append(predicted_per_module_list)
        yield batch, predicted_list


def _predict(tl, module, datamodule, use_sigmoid):
    predicted = tl.predict(module, datamodule)
    with torch.no_grad():
        predicted = torch.concat(predicted).float()
        if use_sigmoid:
            predicted = predicted.sigmoid()
        predicted = predicted.numpy()
    return predicted
