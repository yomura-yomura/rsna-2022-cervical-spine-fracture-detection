import sys
import CSFD.data.three_dimensions

if len(sys.argv) == 1:
    yaml_path = "resnet10.yaml"
else:
    yaml_path = sys.argv[1]

cfg_dataset = CSFD.data.load_yaml_config(yaml_path).dataset


cfg_dataset.depth = 128
cfg_dataset.image_2d_shape = [256, 256]
cfg_dataset.save_images_with_specific_depth = False
CSFD.data.three_dimensions.get_df(cfg_dataset)

cfg_dataset.depth = 128
cfg_dataset.image_2d_shape = [256, 256]
cfg_dataset.save_images_with_specific_depth = True
CSFD.data.three_dimensions.get_df(cfg_dataset)

cfg_dataset.depth = 200
cfg_dataset.image_2d_shape = [256, 256]
cfg_dataset.save_images_with_specific_depth = True
CSFD.data.three_dimensions.get_df(cfg_dataset)

cfg_dataset.depth = 80
cfg_dataset.image_2d_shape = [256, 256]
cfg_dataset.save_images_with_specific_depth = True
CSFD.data.three_dimensions.get_df(cfg_dataset)

cfg_dataset.depth = 128
cfg_dataset.image_2d_shape = [256, 256]
cfg_dataset.save_images_with_specific_depth = True
CSFD.data.three_dimensions.get_df(cfg_dataset)