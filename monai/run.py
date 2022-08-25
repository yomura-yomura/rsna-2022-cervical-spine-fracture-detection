import CSFD.monai
import CSFD.data

cfg = CSFD.data.load_yaml_config("resnet10.yaml")
# module = CSFD.monai.Module(cfg)

CSFD.data.get_df(cfg)
