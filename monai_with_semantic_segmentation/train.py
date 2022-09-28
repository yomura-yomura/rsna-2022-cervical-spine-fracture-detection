import CSFD.data.io.three_dimensions
import CSFD.monai.training


if __name__ == "__main__":
    cfg = CSFD.data.load_yaml_config("SEResNext50.yaml")
    df = CSFD.data.io.three_dimensions.get_df(cfg.dataset)

    # df = pd.merge(semantic_segmentation_bb_df, df, how="inner", on="StudyInstanceUID")
    # CSFD.bounding_box.get_3d_bounding_box()
    datamodule = CSFD.monai.datamodule.CSFDCropped3DDataModule(cfg, df)
    datamodule.setup("fit")

    CSFD.monai.training.train(
        cfg,
        module_class=CSFD.monai.module.CSFDCropped3DModule,
        datamodule_class=CSFD.monai.datamodule.CSFDCropped3DDataModule
    )

