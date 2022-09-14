import CSFD.data.three_dimensions
import CSFD.data.io_with_cfg


cfg = CSFD.data.load_yaml_config("UNet.yaml")
# cfg.dataset.type_to_load = "npz"
# cfg.dataset.image_2d_shape = None
# cfg.dataset.depth = None
# cfg.dataset.depth_range = None
# cfg.dataset.height_range = None
# cfg.dataset.width_range = None
# cfg.dataset.save_images_with_specific_depth = False
# cfg.dataset.save_images_with_specific_height = False
# cfg.dataset.save_images_with_specific_width = False
# cfg.dataset.use_segmentations = True
# cfg.train.augmentation = {}


df = CSFD.data.three_dimensions.get_df(cfg.dataset)
# df = df.dropna()

# import CSFD.monai
# module = CSFD.monai.CSFDModule(cfg)
# datamodule = CSFD.monai.CSFDDataModule(cfg, df)
# datamodule.setup("fit")
# # print(datamodule.train_dataset[0])
# batch = datamodule.train_dataset[0]
# images = batch["segmentation"].numpy()

print(cfg)

import CSFD.monai.training
CSFD.monai.training.train(cfg)


# https://pytorch.org/hub/ultralytics_yolov5/
# model = torch.hub.load('ultralytics/color5', 'yolov5s', autoshape=False, pretrained=False)

