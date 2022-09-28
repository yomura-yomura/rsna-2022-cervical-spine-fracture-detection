import os.path

import omegaconf

import CSFD.data
import CSFD.monai.training
import sys
import optuna
import pathlib
import CSFD.monai.from_checkpoint
import CSFD.data.io.three_dimensions


def objective(trial: optuna.Trial):
    cfg = CSFD.data.load_yaml_config(yaml_path)
    cfg.dataset.type_to_load = "npz"
    cfg.dataset.cv.fold = 0
    cfg.train.name_prefix = f"optuna-trial{trial.number:>03d}-"

    cfg.train.learning_rate = trial.suggest_loguniform("learning_rate", 1e-8, 1e-4)
    cfg.train.weight_decay = trial.suggest_loguniform("weight_decay", 1e-12, 10)
    cfg.model.optimizer.scheduler.name = trial.suggest_categorical(
        "scheduler_name", [None, "linear", "cosine"]
    )
    if cfg.model.optimizer.scheduler.name is not None:
        cfg.model.optimizer.scheduler.num_warmup_steps = trial.suggest_int("num_warmup_steps", 0, 2100)
        cfg.model.optimizer.scheduler.num_training_steps = trial.suggest_int(
            "num_training_steps",
            cfg.model.optimizer.scheduler.num_warmup_steps, 2100
        )

    model_path = (
            pathlib.Path(cfg.train.model_path)
            / "optuna"
            / f"{cfg.model.name}_folds{cfg.dataset.cv.n_folds}_{cfg.train.name_suffix}"
            / f"trial{trial.number:>03d}"
    )
    model_path.mkdir(exist_ok=False, parents=True)
    omegaconf.OmegaConf.save(cfg, model_path / os.path.basename(yaml_path))
    cfg.train.model_path = str(model_path)

    print(cfg)
    CSFD.monai.training.train(cfg)

    cfg, ckpt_dict = CSFD.monai.from_checkpoint.load_cfg_and_checkpoints(model_path)
    df = CSFD.data.io.three_dimensions.get_df(cfg.dataset)
    return CSFD.monai.from_checkpoint.validate(cfg, ckpt_dict[cfg.dataset.cv.fold], df)


db_path = pathlib.Path("model-hyper-params.db")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # yaml_path = "resnet10.yaml"
        # yaml_path = "resnet50.yaml"
        yaml_path = "effnet-bn.yaml"
    else:
        yaml_path = sys.argv[1]

    print(yaml_path)

    study = optuna.create_study(
        storage=f"sqlite:///{db_path}",
        study_name=yaml_path, load_if_exists=True,
        direction="minimize"
    )

    import pandas as pd
    from sqlite3 import connect
    conn = connect(db_path)
    study_id = pd.read_sql("select studies.study_id from studies", conn)["study_id"]

    if yaml_path not in study_id:
        initial_guess = {
            "lr": 1e-4,
            "weight_decay": 1e-12,
            "schedular_name": None
        }
        print(f"Info: Set initial guess as {initial_guess}")
        study.enqueue_trial(initial_guess)

    study.optimize(objective, n_trials=1000)
