import pathlib
import pandas as pd
import CSFD.monai.from_checkpoint
import CSFD.data.three_dimensions
import numpy as np
import sys
import tqdm


if __name__ == "__main__":
    if len(sys.argv) == 1:
        model_path = "models2"
    else:
        model_path = sys.argv[1]

    cfg, ckpt_dict = CSFD.monai.from_checkpoint.load_cfg_and_checkpoints(model_path)
    # cfg.dataset.type_to_load = "dcm"
    cfg.dataset.type_to_load = "npz"
    # cfg.dataset.type = "test"
    cfg.dataset.type = "train"
    cfg.dataset.use_segmentations = False
    # cfg.dataset.test_batch_size = 4

    df = CSFD.data.three_dimensions.get_df(cfg.dataset, ignore_invalids=False)
    assert len(df) == 2019

    output_path = pathlib.Path("predicted_data") / "float16"
    output_path.mkdir(exist_ok=True, parents=True)

    cfg.dataset.use_segmentations = True

    # import pytorch_lightning
    # pytorch_lightning.seed_everything(cfg.model.seed)
    for fold, ckpt_path in ckpt_dict.items():
        print(f"* fold {fold}")

        target_dir = output_path / f"fold{fold}"
        if target_dir.exists():
            print(f"[Info] Skipped {target_dir}")
            continue

        target_dir.mkdir(exist_ok=True)

        # if target_csv.exists():
        #     continue
        cfg.dataset.cv.fold = fold

        from pytorch_lightning import Trainer
        import torch
        tl = Trainer(
            accelerator="gpu", devices=1,
            max_epochs=1000,
            precision=cfg.train.precision
        )
        module = CSFD.monai.CSFDModule.load_from_checkpoint(str(ckpt_path), cfg=cfg, map_location=torch.device("cuda"))

        for i in tqdm.trange(int(np.ceil(len(df) / 10))):
            target_df = df.iloc[10 * i: 10 * (i + 1)]
            datamodule = CSFD.monai.CSFDDataModule(cfg, target_df)
            predicted = tl.predict(module, datamodule)
            with torch.no_grad():
                predicted = torch.concat(predicted).float().sigmoid().numpy()
            predicted = predicted.astype(np.float16)

            assert len(predicted) == len(target_df)
            for data, uid in zip(predicted, target_df["StudyInstanceUID"]):
                np.savez_compressed(target_dir / f"{uid}.npz", data)
