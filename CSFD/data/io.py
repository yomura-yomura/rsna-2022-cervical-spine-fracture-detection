import pandas as pd
import sklearn.model_selection
import omegaconf
import pathlib
import numpy as np


__all__ = ["load_yaml_config", "get_df"]


_folds_csv_root_path = pathlib.Path(__file__).resolve().parent / "_folds_csv"


def load_yaml_config(path):
    cfg = omegaconf.OmegaConf.load(path)
    _validate_cfg(cfg)
    return cfg


def _validate_cfg(cfg):
    cfg.dataset.data_root_path = pathlib.Path(cfg.dataset.data_root_path)


def get_df(cfg):
    df = pd.read_csv(cfg.dataset.data_root_path / f"{cfg.dataset.type}.csv")

    if cfg.dataset.type == "train":
        _folds_csv_path = (
            _folds_csv_root_path / "_".join([
                f"{cfg.dataset.cv.type}",
                f"nFolds{cfg.dataset.cv.n_folds}",
                f"Seed{cfg.dataset.cv.seed}"
            ])
        ).with_suffix(".csv")

        if _folds_csv_path.exists():
            df = pd.concat([df, pd.read_csv(_folds_csv_path)], axis=1)
        else:
            if cfg.dataset.cv.type == "KFold":
                kf = sklearn.model_selection.KFold(
                    n_splits=cfg.dataset.cv.n_folds,
                    shuffle=True, random_state=cfg.dataset.cv.seed
                )
            else:
                raise NotImplementedError(f"Unexpected cfg.dataset.cv.type: {cfg.dataset.cv.type}")

            df["fold"] = -1
            for fold, (_, valid_indices) in enumerate(
                    kf.split(
                        df.drop(columns=cfg.dataset.target_columns),
                        df[cfg.dataset.target_columns]
                    )
            ):
                df.loc[valid_indices, "fold"] = fold
            assert np.all(df["fold"] >= 0)
            df["fold"].to_csv(_folds_csv_path, index=False)
    elif cfg.dataset.type == "test":
        pass
    else:
        raise ValueError(f"Unexpected cfg.dataset.type: {cfg.dataset.type}")

    return df
