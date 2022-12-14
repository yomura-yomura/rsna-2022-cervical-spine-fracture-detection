import omegaconf
import warnings


def load_yaml_config(path, add_default_values=True, show_warning=True):
    cfg = omegaconf.OmegaConf.load(path)
    omegaconf.OmegaConf.set_struct(cfg, True)
    if add_default_values:
        add_default_values_if_not_defined(cfg, show_warning)
    return cfg


def add_default_values_if_not_defined(cfg, show_warnings=True):
    def _validate_cfg(cfg, key, default_value):
        if not hasattr(cfg, key):
            if show_warnings:
                warnings.warn(
                    f"Given cfg does not have key '{key}'. Tt will be given with default value '{default_value}'",
                    UserWarning
                )
            with omegaconf.open_dict(cfg):
                setattr(cfg, key, default_value)
    def _update_recursively_if_not_defined(cfg: dict, base_cfg: dict):
        for k, v in base_cfg.items():
            if not hasattr(cfg, k):
                continue
            if getattr(cfg, k, None) is None:
                continue
            if not isinstance(getattr(cfg, k), dict):
                assert isinstance(v, dict)
                _update_recursively_if_not_defined(getattr(cfg, k), v)
                continue
            setattr(cfg, k, v)

    # Needed just for compatibility
    for cfg_key, default_map_dict in {
        "dataset": {
            "type_to_load": "dcm",
            "enable_depth_resized_with_cv2": True,
            "data_type": "f4",

            "train_3d_images": None,
            "train_cropped_3d_images_path": None,

            "image_2d_shape": None,
            "depth": None,
            "depth_range": None,
            "height_range": None,
            "width_range": None,
            "save_images_with_specific_depth": False,
            "save_images_with_specific_height": False,
            "save_images_with_specific_width": False,

            "use_normalized_batches": True,
            "equalize_adapthist": False,

            "use_segmentation": False,
            "train_segmentations_path": None,
            "semantic_segmentation_bb_path": None,

            "cv": None,
            "num_workers": None,

            "use_voi_lut": False,
            "use_windowing": False,

            "train_cache_rate": 0,
            "valid_cache_rate": 0,

            "cropped_2d_labels_path": None
        },
        "model": {
            "spatial_dims": 3,
            "use_multi_sample_dropout": False,
            "use_medical_net": False,
            "optimizer": {
                "scheduler": {
                    "name": None
                }
            }
        },
        "train": {
            "seed": 42,
            "early_stopping": False,
            "save_on_train_epoch_end": False,
            "augmentation": None,
            "pos_weight": None
        }
    }.items():
        if not hasattr(cfg, cfg_key):
            if show_warnings:
                warnings.warn(f"cfg doesn't have a key '{cfg_key}'")
            continue
        cfg_key = getattr(cfg, cfg_key)
        for key, default_value in default_map_dict.items():
            _validate_cfg(cfg_key, key, default_value)

    if hasattr(cfg, "model"):
        # cfg.model.optimizer.scheduler
        if not hasattr(cfg.model.optimizer.scheduler, "kwargs"):
            with omegaconf.open_dict(cfg):
                cfg.model.optimizer.scheduler.kwargs = dict()
        if hasattr(cfg.model.optimizer.scheduler, "num_warmup_steps"):
            with omegaconf.open_dict(cfg):
                cfg.model.optimizer.scheduler.kwargs["num_warmup_steps"] = cfg.model.optimizer.scheduler.num_warmup_steps
        if hasattr(cfg.model.optimizer.scheduler, "num_training_steps"):
            with omegaconf.open_dict(cfg):
                cfg.model.optimizer.scheduler.kwargs["num_training_steps"] = cfg.model.optimizer.scheduler.num_training_steps

        # cfg.model.name
        if cfg.model.name.startswith("resnet"):
            with omegaconf.open_dict(cfg):
                if not hasattr(cfg.model.kwargs, "n_input_channels"):
                    cfg.model.kwargs = dict(n_input_channels=1)

    return cfg
