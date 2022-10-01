import sys
import CSFD.data.io.three_dimensions


if len(sys.argv) == 1:
    yaml_path = "resnet10.yaml"
else:
    yaml_path = sys.argv[1]

cfg_dataset = CSFD.data.io.load_yaml_config(yaml_path).dataset

cfg_dataset.type_to_load = "npz"

# cfg_dataset.depth = 128
# cfg_dataset.save_images_with_specific_depth = False
# cfg_dataset.image_2d_shape = [256, 256]
# cfg_dataset.height_range = [0.1, 0.8]
# cfg_dataset.width_range = [0.15, 0.85]
# cfg_dataset.save_images_with_specific_height = True
# cfg_dataset.save_images_with_specific_width = True
cfg_dataset.depth = None
cfg_dataset.depth_range = None
cfg_dataset.image_2d_shape = None
cfg_dataset.height_range = None
cfg_dataset.width_range = None
cfg_dataset.use_windowing = True

df = CSFD.data.io_with_cfg.three_dimensions.get_df(cfg_dataset)


import numpy as np
import tqdm

for npz in tqdm.tqdm(df["np_images_path"]):
    np.load(npz)["arr_0"]

# cfg_dataset.depth = 128
# cfg_dataset.image_2d_shape = [256, 256]
# cfg_dataset.save_images_with_specific_depth = True
# CSFD.data.three_dimensions.get_df(cfg_dataset)
#
# cfg_dataset.depth = 200
# cfg_dataset.image_2d_shape = [256, 256]
# cfg_dataset.save_images_with_specific_depth = True
# CSFD.data.three_dimensions.get_df(cfg_dataset)
#
# cfg_dataset.depth = 80
# cfg_dataset.image_2d_shape = [256, 256]
# cfg_dataset.save_images_with_specific_depth = True
# CSFD.data.three_dimensions.get_df(cfg_dataset)
#
# cfg_dataset.depth = 128
# cfg_dataset.image_2d_shape = [256, 256]
# cfg_dataset.save_images_with_specific_depth = True
# CSFD.data.three_dimensions.get_df(cfg_dataset)
