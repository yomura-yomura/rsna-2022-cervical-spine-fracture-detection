import CSFD.data.three_dimensions
import CSFD.bounding_box


if __name__ == "__main__":
    cfg = CSFD.data.load_yaml_config("SEResNext50.yaml")
    df = CSFD.data.three_dimensions.get_df(cfg.dataset)

    CSFD.bounding_box.get_3d_bounding_box()