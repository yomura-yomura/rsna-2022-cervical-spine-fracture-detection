import pathlib
import pandas as pd
import CSFD.monai.from_checkpoint
import CSFD.data.three_dimensions
import CSFD.metric
import numpy as np
import sys


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # model_path = "models/resnet10_folds5_test-v3"
        # model_path = "models/resnet10_folds2_test-v1.0"
        model_path = "models/resnet10_folds4_test-v3.2"
    else:
        model_path = sys.argv[1]

    cfg, ckpt_dict = CSFD.monai.from_checkpoint.load_cfg_and_checkpoints(model_path)
    cfg.dataset.type_to_load = "npz"

    predicted_csv_paths = {
        int(p.name[4:-4]): p
        for p in pathlib.Path(model_path).glob(f"predicted_csv/fold*.csv")
    }

    cv_list = []
    for fold in range(cfg.dataset.cv.n_folds):
        if fold in predicted_csv_paths.keys():
            predicted_csv_path = predicted_csv_paths[fold]
            df = CSFD.data.three_dimensions.get_df(cfg.dataset, ignore_invalids=False)
            predicted_df = pd.read_csv(predicted_csv_path)
            df, predicted_df = CSFD.data.io.drop_invalids(df, predicted_df)
            predicted = predicted_df[df["fold"] == fold].values
            true = df.loc[df["fold"] == fold, list(cfg.dataset.target_columns)].values
            # predicted[:, 0] = 1 - np.prod(1 - predicted[:, 1:], axis=1)
            cv_list.append(
                CSFD.metric.numpy.competition_loss(predicted, true)
            )
        else:
            df = CSFD.data.three_dimensions.get_df(cfg.dataset)
            cv_list.append(
                CSFD.monai.from_checkpoint.validate(cfg, ckpt_dict[fold], df)
            )

    cv_str = " ".join(map("{:.2f}".format, cv_list))
    print(f"CV: {np.mean(cv_list):.2f} ({cv_str})")
