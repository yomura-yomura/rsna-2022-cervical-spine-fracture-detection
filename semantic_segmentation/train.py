import omegaconf

import CSFD.data.io.three_dimensions
import CSFD.monai.training
import sklearn.model_selection


if __name__ == "__main__":
    cfg = CSFD.data.io.load_yaml_config("UNet.yaml")
    df = CSFD.data.io_with_cfg.three_dimensions.get_df(cfg.dataset)

    assert cfg.dataset.cv is None
    n_folds = 7
    skf = sklearn.model_selection.StratifiedKFold(n_splits=n_folds)
    for fold, (_, valid_idx) in enumerate(skf.split(df, df["patient_overall"])):
        df.loc[df.index[valid_idx], "fold"] = fold
    with omegaconf.open_dict(cfg):
        cfg.dataset.cv = dict(
            fold=1,
            # fold=[2, 3, 4, 5, 6],
            n_folds=n_folds
        )

    # df["fold"] = -1
    # cfg.dataset.cv.fold = 0

    print(cfg)
    CSFD.monai.training.train(
        cfg,
        module_class=CSFD.monai.module.CSFDSemanticSegmentationModule,
        datamodule_class=CSFD.monai.datamodule.CSFDSemanticSegmentationDataModule,
        df=df
    )
