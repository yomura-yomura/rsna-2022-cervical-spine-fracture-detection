from ..io import two_dimensions as _io_two_dimensions_module


def get_df(dataset_cfg, ignore_invalid=True):
    return _io_two_dimensions_module.get_df(
        dataset_cfg.data_root_path, dataset_cfg.type,
        dataset_cfg.cv, dataset_cfg.target_columns,
        ignore_invalid=ignore_invalid
    )
