import CSFD.data.io.three_dimensions
import CSFD.monai.training


if __name__ == "__main__":
    cfg = CSFD.data.io.load_yaml_config("SEResNext50.yaml")

    # df = CSFD.data.io_with_cfg.three_dimensions.get_df(cfg.dataset)
    # df = pd.merge(semantic_segmentation_bb_df, df, how="inner", on="StudyInstanceUID")
    # CSFD.bounding_box.get_3d_bounding_box()
    # datamodule = CSFD.monai.datamodule.CSFDCropped3DDataModule(cfg, df)
    # datamodule.setup("fit")
    # loader = datamodule.train_dataloader()
    # print("start")
    # data = next(iter(loader))
    # cfg.dataset.cv.fold = 0
    # datamodule = CSFD.monai.datamodule.CSFDCropped3DDataModule(cfg, df)
    # datamodule.setup("fit")
    # datamodule.setup("predict")

    CSFD.monai.training.train(
        cfg,
        module_class=CSFD.monai.module.CSFDCroppedModule,
        datamodule_class=CSFD.monai.datamodule.CSFDCropped3DDataModule
    )
