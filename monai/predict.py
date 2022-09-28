import pathlib
import pandas as pd
import CSFD.monai.from_checkpoint
import CSFD.data.io.three_dimensions
import sys

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # model_path = "models/resnet10_folds5_test-v3"
        # model_path = "models/resnet10_folds2_test-v1.0"
        # model_path = "models/resnet10_folds4_test-v3.2"
        # model_path = "models/EfficientNetBN_folds4_test-v4.2"
        model_path = "models/EfficientNetBN_folds4_test-v4.3"
    else:
        model_path = sys.argv[1]

    cfg, ckpt_dict = CSFD.monai.from_checkpoint.load_cfg_and_checkpoints(model_path)
    # cfg.dataset.type_to_load = "dcm"
    cfg.dataset.type_to_load = "npz"
    # cfg.dataset.type = "test"
    cfg.dataset.type = "train"
    # cfg.dataset.test_batch_size = 4

    df = CSFD.data.io.three_dimensions.get_df(cfg.dataset, ignore_invalids=False)
    assert len(df) == 2019
    # fdsaf
    # predicted_dict = CSFD.monai.from_checkpoint.predict_all_folds(cfg, ckpt_dict, df)

    output_path = pathlib.Path(model_path) / "predicted_csv"
    output_path.mkdir(exist_ok=True)

    # import pytorch_lightning
    # pytorch_lightning.seed_everything(cfg.model.seed)
    for fold, ckpt_path in ckpt_dict.items():
        target_csv = output_path / f"fold{fold}.csv"
        # if target_csv.exists():
        #     continue
        print(f"* fold {fold}")
        cfg.dataset.cv.fold = fold
        predicted = CSFD.monai.from_checkpoint.predict(cfg, ckpt_path, df)
        print(predicted)
        predicted_df = pd.DataFrame(
            predicted, columns=cfg.dataset.target_columns
        )
        predicted_df["StudyInstanceUID"] = df["StudyInstanceUID"]
        predicted_df = predicted_df[["StudyInstanceUID", *cfg.dataset.target_columns]]
        predicted_df.to_csv(target_csv, index=False)
