import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '/'))
import CSFD.data.three_dimensions


if len(sys.argv) == 1:
    yaml_path = "resnet10_make.yaml"
else:
    yaml_path = sys.argv[1]

cfg_dataset = CSFD.data.load_yaml_config(yaml_path).dataset

cfg_dataset.type_to_load = "npz"

cfg_dataset.depth = 256
cfg_dataset.image_2d_shape = [512, 512]
cfg_dataset.save_images_with_specific_depth = False

cfg_dataset.height_range = [0.1, 0.8]
cfg_dataset.width_range = [0.15, 0.85]
cfg_dataset.save_images_with_specific_height = True
cfg_dataset.save_images_with_specific_width = True

df = CSFD.data.three_dimensions.get_df(cfg_dataset)


# import numpy as np
# import tqdm
#
# for npz in tqdm.tqdm(df["np_images_path"]):
#     np.load(npz)["arr_0"]

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
