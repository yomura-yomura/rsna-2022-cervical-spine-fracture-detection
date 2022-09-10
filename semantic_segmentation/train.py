import CSFD.data.three_dimensions
import CSFD.data.io_with_cfg


cfg = CSFD.data.load_yaml_config("../monai/resnet10.yaml")
cfg.dataset.type_to_load = "npz"
cfg.dataset.image_2d_shape = None
cfg.dataset.depth = None
cfg.dataset.depth_range = None
cfg.dataset.height_range = None
cfg.dataset.width_range = None
cfg.dataset.save_images_with_specific_depth = False
cfg.dataset.save_images_with_specific_height = False
cfg.dataset.save_images_with_specific_width = False
cfg.dataset.use_segmentations = True
cfg.train.augmentation = {}


df = CSFD.data.three_dimensions.get_df(cfg.dataset)
df = df.dropna()


for nil_path, dcm_path in zip(df["nil_images_path"], df["dcm_images_path"]):
    break



# https://pytorch.org/hub/ultralytics_yolov5/
# model = torch.hub.load('ultralytics/color5', 'yolov5s', autoshape=False, pretrained=False)

