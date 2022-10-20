import CSFD.data.io.three_dimensions
import CSFD.monai.training


if __name__ == "__main__":
    cfg = CSFD.data.io.load_yaml_config("SEResNext50.yaml")
    df = CSFD.data.io_with_cfg.three_dimensions.get_df(cfg.dataset)

    # # cfg.dataset.semantic_segmentation_bb_path = "../semantic_segmentation/models/UNet_128x256x256/semantic_segmentation_bb/train_semantic_segmentation_bb_fold0.csv"
    # # cfg.dataset.cropped_2d_images_path = "../data/cropped_2d_images/UNet_128x256x256"
    #
    # # cfg.dataset.num_workers = 0
    #
    # # cfg.dataset.cv.fold = 0
    # # datamodule = CSFD.monai.datamodule.CSFDCropped2DDataModule(cfg, df)
    # # datamodule.setup("fit")
    # # print(len(datamodule.train_dataset), len(datamodule.valid_dataset))
    # datamodule = CSFD.monai.datamodule.CSFDCropped2DDataModule(cfg, df)
    # datamodule.setup("predict")
    # datamodule.test_dataset.save()
    # # datamodule = CSFD.monai.datamodule.CSFDCropped2DDataModule(cfg, df)
    # # datamodule.setup("predict")
    #
    #
    # dataset = datamodule.test_dataset
    # ss_bb_df = dataset.semantic_segmentation_bb_df.reset_index(drop=True)
    #
    # uid = "1.2.826.0.1.3680043.10051"

    # import plotly.express as px
    # import plotly_utility
    # plotly_utility.offline.mpl_plot(px.imshow(datamodule.train_dataset[8]["data"], color_continuous_scale="gray"))


    # plotly_utility.offline.mpl_plot(
    #     px.imshow(datamodule.train_dataset[27074]["data"][0], color_continuous_scale="gray")
    # )

    # loader = datamodule.train_dataloader()
    # print("start")
    # data = next(iter(loader))

    cfg.dataset.num_workers = None
    CSFD.monai.training.train(
        cfg,
        module_class=CSFD.monai.module.CSFDCroppedModule,
        datamodule_class=CSFD.monai.datamodule.CSFDCropped2DDataModule
    )

