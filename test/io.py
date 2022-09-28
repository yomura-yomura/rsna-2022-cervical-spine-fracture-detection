import CSFD.data.io.three_dimensions
import attrdict

dataset_cfg = attrdict.AttrDict(dict(
    type="train",
    data_root_path="../data/rsna-2022-cervical-spine-fracture-detection",
    train_3d_images="../data/3d_train_images_v2",

    data_type="f4",

    enable_depth_resized_with_cv2=True,

    depth_range=[0.1, 0.9],
    height_range=[0.2, 0.8],
    width_range=[0.2, 0.8],

    image_2d_shape=None,
    depth=None,

    save_images_with_specific_depth=False,

    type_to_load="dcm",

    cv=dict(
        type="KFold",
        seed=42,
        n_folds=4,
        fold=None
    )
))

df = CSFD.data.io.three_dimensions.get_df(dataset_cfg)


dcm_images_path = df["dcm_images_path"].iloc[0]
images = CSFD.data.io.three_dimensions.load_3d_images(dcm_images_path)

