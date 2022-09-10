import CSFD.data.three_dimensions
from CSFD.monai.transforms import LoadImage
import attrdict
from monai.data import Dataset, DataLoader


# dataset_cfg = attrdict.AttrDict(dict(
#     type="train",
#     data_root_path="../data/rsna-2022-cervical-spine-fracture-detection",
#     train_3d_images="../data/3d_train_images_v2",
#
#     data_type="f4",
#
#     enable_depth_resized_with_cv2=True,
#
#     depth_range=[0.1, 0.9],
#     height_range=[0.2, 0.8],
#     width_range=[0.2, 0.8],
#
#     image_2d_shape=None,
#     depth=None,
#
#     save_images_with_specific_depth=False,
#
#     type_to_load="dcm",
#
#     target_columns=['patient_overall', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'],
#     use_normalized_batches=True,
#
#     cv=dict(
#         type="KFold",
#         seed=42,
#         n_folds=4,
#         fold=None
#     )
# ))
#
# df = CSFD.data.three_dimensions.get_df(dataset_cfg)


cfg = CSFD.data.load_yaml_config("../monai/resnet10.yaml")

df = CSFD.data.three_dimensions.get_df(cfg.dataset)
cfg.dataset.type_to_load = "npz"

transforms = CSFD.monai.transforms.get_transforms(cfg, is_train=True)

dataset = Dataset(df.to_dict("records"), transforms)
ret = dataset[0]
print(ret.keys())
print(ret["data"].shape)

# dataloader = DataLoader(dataset)
