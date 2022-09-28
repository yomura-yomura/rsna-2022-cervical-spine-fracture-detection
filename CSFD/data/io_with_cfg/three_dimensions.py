import numpy as np
import warnings
import CSFD.data.io.three_dimensions
import skimage.exposure




def get_df(cfg_dataset, ignore_invalids=True, n_jobs_to_save_images=-1):
    return CSFD.data.io.three_dimensions.get_df(
        cfg_dataset.data_root_path, cfg_dataset.type, 
        cfg_dataset.type_to_load,
        cfg_dataset.train_3d_images, cfg_dataset.data_type,
        cfg_dataset.depth, cfg_dataset.depth_range, cfg_dataset.save_images_with_specific_depth,
        cfg_dataset.image_2d_shape,
        cfg_dataset.height_range, cfg_dataset.save_images_with_specific_height,
        cfg_dataset.width_range, cfg_dataset.save_images_with_specific_width,
        cfg_dataset.enable_depth_resized_with_cv2,
        
        cfg_dataset.cv, cfg_dataset.target_columns,
        cfg_dataset.use_segmentations, cfg_dataset.train_segmentations_path,

        ignore_invalids=ignore_invalids, n_jobs_to_save_images=n_jobs_to_save_images
    )



def load_3d_images(image_path, cfg_dataset):
    if cfg_dataset.type_to_load in "npz":
        images = np.load(image_path)["arr_0"]
        if np.issubdtype(images.dtype, np.uint8):
            images = images.astype("f4")
        elif np.issubdtype(images.dtype, np.float32):
            pass
        else:
            warnings.warn(f"not expected type: {images.dtype}")

        if len(images) != cfg_dataset.depth:
            images = CSFD.data.io.three_dimensions.resize_depth(
                images,
                cfg_dataset.depth, cfg_dataset.depth_range,
                cfg_dataset.enable_depth_resized_with_cv2
            )
    elif cfg_dataset.type_to_load == "dcm":
        images = CSFD.data.io.three_dimensions.load_3d_images(
            image_path,
            cfg_dataset.image_2d_shape,
            cfg_dataset.enable_depth_resized_with_cv2,
            cfg_dataset.data_type,
            depth=cfg_dataset.depth, depth_range=cfg_dataset.depth_range,
            n_jobs=1,
            voi_lut=cfg_dataset.use_voi_lut, widowing=cfg_dataset.use_windowing
        )
    else:
        raise RuntimeError

    if cfg_dataset.equalize_adapthist:
        # https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_adapt_hist_eq_3d.html
        images /= images.max()
        images = skimage.exposure.equalize_adapthist(images, kernel_size=np.array([64, 64, 64]), clip_limit=0.01)
        images *= 255
        images = np.clip(images, 0, 255)
    # assert vol.dtype == np.dtype(cfg_dataset.data_type)

    images = images[np.newaxis, ...]
    if cfg_dataset.use_normalized_batches:
        # images -= images.mean()
        # images /= images.std()
        images /= 255

    return images
