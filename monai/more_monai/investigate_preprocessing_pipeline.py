import monai.transforms as mt
import CSFD.data.io.three_dimensions

cfg = CSFD.data.load_yaml_config("../resnet10.yaml")

cfg.dataset.data_root_path = f"../{cfg.dataset.data_root_path}"
cfg.dataset.type_to_load = "dcm"
df = CSFD.data.io.three_dimensions.get_df(cfg.dataset)


transforms = mt.Compose([
    mt.LoadImage(reader="PydicomReader"),
    # mt.CenterScaleCrop(roi_scale=[0.5, 0.5, 0.5]),
    # mt.Resize(mode="trilinear", spatial_size=[124, 100, 100])
])

import tqdm
r1 = [transforms(p) for p in tqdm.tqdm(df["dcm_images_path"].iloc[:10])]
r2 = [CSFD.data.io.three_dimensions.load_3d_images(p) for p in tqdm.tqdm(df["dcm_images_path"].iloc[:10])]