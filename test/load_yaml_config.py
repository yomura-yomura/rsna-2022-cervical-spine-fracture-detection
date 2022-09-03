import CSFD.data
import pprint

cfg = CSFD.data.load_yaml_config("../monai/models/resnet10_folds5_test-v3/resnet10.yaml")
pprint.pprint(dict(cfg))
