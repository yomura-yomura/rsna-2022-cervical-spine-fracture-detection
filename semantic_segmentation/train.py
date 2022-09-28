import CSFD.data.three_dimensions
import CSFD.monai.training

if __name__ == "__main__":
    cfg = CSFD.data.load_yaml_config("UNet.yaml")
    df = CSFD.data.three_dimensions.get_df(cfg.dataset)

    print(cfg)
    CSFD.monai.training.train(
        cfg,
        module_class=CSFD.monai.module.CSFDSemanticSegmentationModule
    )