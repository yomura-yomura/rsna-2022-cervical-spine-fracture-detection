import CSFD.data.io.three_dimensions
import CSFD.monai.training

if __name__ == "__main__":
    cfg = CSFD.data.io.load_yaml_config("UNet.yaml")
    df = CSFD.data.io_with_cfg.three_dimensions.get_df(cfg.dataset)

    print(cfg)
    CSFD.monai.training.train(
        cfg,
        module_class=CSFD.monai.module.CSFDSemanticSegmentationModule
    )
