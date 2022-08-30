import pathlib
import pandas as pd
import torch

import CSFD.monai.from_checkpoint
import CSFD.data.three_dimensions
import CSFD.metric
import numpy as np


model_path = "models/resnet10_folds5_test-v3"
# from_scratch = True
from_scratch = False


cfg, ckpt_dict = CSFD.monai.from_checkpoint.load_cfg_and_checkpoints(model_path)
cfg.dataset.type_to_load = "npz"
df = CSFD.data.three_dimensions.get_df(cfg.dataset)

if from_scratch:
    # df = pd.concat([df[df["fold"] == fold].iloc[:10] for fold in df["fold"].unique()])
    cv_dict = CSFD.monai.from_checkpoint.validate_all_folds(cfg, ckpt_dict, df)

    cv_list = [cv_dict[fold] for fold in np.arange(cfg.dataset.cv.n_folds)]
else:
    cv_list = []
    for fold in range(cfg.dataset.cv.n_folds):
        predicted = pd.read_csv(pathlib.Path(model_path) / "predicted_csv" / f"fold{fold}.csv")[df["fold"] == fold].values
        true = df.loc[df["fold"] == fold, list(cfg.dataset.target_columns)].values
        # predicted[:, 0] = 1 - np.prod(1 - predicted[:, 1:], axis=1)
        cv_list.append(
            CSFD.metric.numpy.competition_loss(predicted, true)
        )

cv_str = " ".join(map("{:.2f}".format, cv_list))
print(f"CV: {np.mean(cv_list):.2f} ({cv_str})")
