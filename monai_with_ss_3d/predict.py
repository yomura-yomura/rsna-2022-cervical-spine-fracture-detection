import CSFD.data.io.three_dimensions
import CSFD.monai.from_checkpoint
import pathlib
import pandas as pd
import numpy as np


if __name__ == "__main__":
    model_path = pathlib.Path("models/normal")
    output_path = model_path / "predicted_csv"
    output_path.mkdir(exist_ok=True)
    # cfg = CSFD.data.io.load_yaml_config("SEResNext50.yaml")
    cfg, checkpoints = CSFD.monai.from_checkpoint.load_cfg_and_checkpoints(model_path)
    cfg.dataset.test_batch_size = 8
    cfg.dataset.num_workers = None
    df = CSFD.data.io_with_cfg.three_dimensions.get_df(cfg.dataset)

    for fold, ckpt_path in checkpoints.items():
        print(f"* fold {fold}")

        target_csv = output_path / f"fold{fold}.csv"
        if target_csv.exists():
            predicted_df = pd.read_csv(target_csv)
        else:
            cfg.dataset.cv.fold = fold
            predicted = CSFD.monai.from_checkpoint.predict(
                cfg, ckpt_path, df,
                module_class=CSFD.monai.module.CSFDCroppedModule, datamodule_class=CSFD.monai.datamodule.CSFDCropped3DDataModule
            )
            predicted = predicted.flatten()
            datamodule = CSFD.monai.datamodule.CSFDCropped3DDataModule(cfg, df)
            datamodule.setup("predict")
            predicted_df = datamodule.test_dataset.label_df.reset_index()
            assert len(predicted_df) == len(predicted)
            predicted_df["fraction"] = predicted
            predicted_df.to_csv(target_csv, index=False)

        # just for validation
        predicted_df["fold"] = predicted_df["fold"].astype(int)
        fold_df = predicted_df[["StudyInstanceUID", "fold"]].drop_duplicates()
        fold_df = fold_df.set_index("StudyInstanceUID").loc[df["StudyInstanceUID"]]
        assert np.all(fold_df["fold"] == df.set_index("StudyInstanceUID")["fold"])

        predicted_df = predicted_df.set_index(["StudyInstanceUID", "type"])["fraction"]
        predicted_df = predicted_df.reindex(pd.MultiIndex.from_product(predicted_df.index.levels))
        predicted_df = predicted_df.unstack()

        print(f"{np.count_nonzero(predicted_df.isna()):,} nan-values of {predicted_df.size:,} is replaced with 0.5")
        predicted_df = predicted_df.fillna(0.5)
        predicted_df["patient_overall"] = predicted_df.values.max(axis=1)
        predicted_df = pd.merge(predicted_df, df["StudyInstanceUID"], on="StudyInstanceUID", how="right")
        predicted = predicted_df[cfg.dataset.target_columns].values
        true = df[cfg.dataset.target_columns].values

        import CSFD.metric.numpy
        sel = df["fold"] == fold
        print(f"train score = {CSFD.metric.numpy.competition_loss(predicted[~sel], true[~sel]):.2f}")
        print(f"valid score = {CSFD.metric.numpy.competition_loss(predicted[sel], true[sel]):.2f}")
