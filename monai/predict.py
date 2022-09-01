import pathlib
import pandas as pd
import CSFD.monai.from_checkpoint
import CSFD.data.three_dimensions
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
    # cfg.dataset.type_to_load = "dcm"
    # cfg.dataset.type = "test"
    cfg.dataset.type = "train"

    df = CSFD.data.three_dimensions.get_df(cfg.dataset, ignore_invalids=False)
    assert len(df) == 2019

    predicted_dict = CSFD.monai.from_checkpoint.predict_all_folds(cfg, ckpt_dict, df)

    output_path = pathlib.Path(model_path) / "predicted_csv"
    output_path.mkdir(exist_ok=True)
    for fold, predicted in predicted_dict.items():
        pd.DataFrame(predicted, columns=cfg.dataset.target_columns).to_csv(output_path / f"fold{fold}.csv", index=False)


if False:
    submission_df = CSFD.data.get_submission_df(cfg.dataset)
    assert np.all(df["row_id"] == submission_df["row_id"])

    predicted = {row_id: None for row_id in df["row_id"]}
    for batch, predicted_list in CSFD.monai.from_checkpoint.predict_on_datamodule_wide(cfg, ckpt_dict, df):
        mean_predicted = np.mean(predicted_list, axis=0)

        assert len(mean_predicted) == len(batch["uid"])
        for row, uid in zip(mean_predicted, batch["uid"]):
            for pred_type, prob in zip(cfg.dataset.target_columns, row):
                predicted["_".join([uid, pred_type])] = prob
