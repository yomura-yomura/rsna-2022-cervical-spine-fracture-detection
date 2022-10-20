import CSFD.data.io.three_dimensions
import CSFD.monai.from_checkpoint
import pathlib
import pandas as pd


if __name__ == "__main__":
    model_path = pathlib.Path("models/test")
    output_path = model_path / "predicted_csv"
    output_path.mkdir(exist_ok=True)
    # cfg = CSFD.data.io.load_yaml_config("SEResNext50.yaml")
    cfg, checkpoints = CSFD.monai.from_checkpoint.load_cfg_and_checkpoints(model_path)
    cfg.dataset.test_batch_size = 64
    df = CSFD.data.io_with_cfg.three_dimensions.get_df(cfg.dataset)
    for fold, ckpt_path in checkpoints.items():
        target_csv = output_path / f"fold{fold}.csv"
        print(f"* fold {fold}")
        cfg.dataset.cv.fold = fold
        predicted = CSFD.monai.from_checkpoint.predict(
            cfg, ckpt_path, df,
            module_class=CSFD.monai.module.CSFDCroppedModule, datamodule_class=CSFD.monai.datamodule.CSFDCropped2DDataModule
        )
        predicted = predicted.flatten()
        datamodule = CSFD.monai.datamodule.CSFDCropped2DDataModule(cfg, df)
        datamodule.setup("predict")
        assert len(datamodule.test_dataset) == len(predicted)

        predicted_df = datamodule.test_dataset.semantic_segmentation_bb_df[
            ["StudyInstanceUID", "type", "slice_number", "count"]
        ].copy()
        predicted_df["fraction"] = predicted

        # predicted_df["StudyInstanceUID"] = df["StudyInstanceUID"]
        # predicted_df = predicted_df[["StudyInstanceUID", *cfg.dataset.target_columns]]
        predicted_df.to_csv(target_csv, index=False)

        # for idx in df.index:
        #     uid = df.loc[idx, "StudyInstanceUID"]
        #     pdf = predicted_df[predicted_df["StudyInstanceUID"] == uid]
        #     break
        # import plotly.express as px
        # import plotly_utility
        # fig = px.histogram(pdf, x="fraction", color="type", barmode="overlay")
        # plotly_utility.offline.mpl_plot(fig)