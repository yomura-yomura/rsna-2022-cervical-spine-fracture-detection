import pathlib
import pandas as pd
import CSFD.monai.from_checkpoint
import CSFD.data.three_dimensions
import numpy as np


model_path = "models/resnet10_folds5_test-v3"

cfg, ckpt_dict = CSFD.monai.from_checkpoint.load_cfg_and_checkpoints(model_path)
# cfg.dataset.type_to_load = "dcm"
# cfg.dataset.type = "test"
cfg.dataset.type_to_load = "npz"
cfg.dataset.type = "train"

df = CSFD.data.three_dimensions.get_df(cfg.dataset)

predicted_dict = CSFD.monai.from_checkpoint.predict_all_folds(cfg, ckpt_dict, df)

output_path = pathlib.Path(model_path) / "predicted_csv"
output_path.mkdir()
for fold, predicted in predicted_dict.items():
    pd.DataFrame(predicted, columns=cfg.dataset.target_columns).to_csv(output_path / f"fold{fold}.csv", index=False)

# predicted = np.mean(list(predicted_dict.values()), axis=0)
#
# predicted_df = df.copy()
# predicted_df.loc[:, cfg.dataset.target_columns] = predicted
#
# submission_df = pd.DataFrame([
#     (f"{row['StudyInstanceUID']}_{target_col}", row[target_col])
#     for _, row in predicted_df.iterrows()
#     for target_col in cfg.dataset.target_columns
# ], columns=["row_id", "fractured"])
# print(submission_df)
fdasf

submission_df = CSFD.data.get_submission_df(cfg.dataset)
assert np.all(df["row_id"] == submission_df["row_id"])

predicted = {row_id: None for row_id in df["row_id"]}
for batch, predicted_list in CSFD.monai.from_checkpoint.predict_on_datamodule_wide(cfg, ckpt_dict, df):
    mean_predicted = np.mean(predicted_list, axis=0)

    assert len(mean_predicted) == len(batch["uid"])
    for row, uid in zip(mean_predicted, batch["uid"]):
        for pred_type, prob in zip(cfg.dataset.target_columns, row):
            predicted["_".join([uid, pred_type])] = prob
