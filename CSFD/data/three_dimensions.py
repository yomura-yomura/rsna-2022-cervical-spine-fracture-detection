import cv2
import pydicom
from PIL import Image
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma
from pydicom.pixel_data_handlers import apply_voi_lut
from kaggle_volclassif.utils import interpolate_volume
from skimage import exposure
import pathlib
import torch


def convert_volume(dir_path, out_dir: str = "test_volumes", size = (224, 224, 224)):
    dir_path = pathlib.Path(dir_path)
    image_paths = sorted(dir_path.glob("*.dcm"))

    imgs = []
    for img_p in image_paths:
        dicom = pydicom.dcmread(img_p)
        img = apply_voi_lut(dicom.pixel_array, dicom)
        img = cv2.resize(img, size[:2], interpolation=cv2.INTER_LINEAR)
        imgs.append(img.tolist())
    vol = torch.tensor(imgs, dtype=torch.float32)

    vol = (vol - vol.min()) / float(vol.max() - vol.min())
    vol = interpolate_volume(vol, size).numpy()

    # https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_adapt_hist_eq_3d.html
    vol = exposure.equalize_adapthist(vol, kernel_size=np.array([64, 64, 64]), clip_limit=0.01)
    # vol = exposure.equalize_hist(vol)
    vol = np.clip(vol * 255, 0, 255).astype(np.uint8)

    path_pt = os.path.join(out_dir, f"{os.path.basename(dir_path)}.pt")
    torch.save(torch.tensor(vol), path_pt)