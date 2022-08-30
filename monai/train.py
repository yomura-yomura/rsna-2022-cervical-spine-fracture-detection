import CSFD.data
import CSFD.monai.training
import sys
import pathlib
import shutil


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # yaml_path = "resnet10.yaml"
        # yaml_path = "resnet50.yaml"
        yaml_path = "effnet-bn.yaml"
    else:
        yaml_path = sys.argv[1]

    print(yaml_path)

    cfg = CSFD.data.load_yaml_config(yaml_path)
    cfg.dataset.type_to_load = "npz"

    model_path = pathlib.Path(
        cfg.train.model_path) / f"{cfg.model.name}_folds{cfg.dataset.cv.n_folds}_{cfg.train.name_suffix}"
    model_path.mkdir(exist_ok=False, parents=True)
    shutil.copy(yaml_path, model_path)

    cfg.train.model_path = model_path
    CSFD.monai.training.train(cfg)
