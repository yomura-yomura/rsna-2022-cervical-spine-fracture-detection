import pathlib
import CSFD.monai.from_checkpoint
import CSFD.data.io.three_dimensions
import numpy as np
import sys
import tqdm


if __name__ == "__main__":
    if len(sys.argv) == 1:
        model_path = "models"
    else:
        model_path = sys.argv[1]

    model_path = pathlib.Path(model_path)

    cfg, ckpt_dict = CSFD.monai.from_checkpoint.load_cfg_and_checkpoints(model_path)
    # cfg.dataset.type_to_load = "dcm"
    cfg.dataset.type_to_load = "npz"
    # cfg.dataset.type = "test"
    cfg.dataset.type = "train"
    cfg.dataset.use_segmentations = False
    # cfg.dataset.test_batch_size = 4

    df = CSFD.data.io_with_cfg.three_dimensions.get_df(cfg.dataset, ignore_invalids=False)
    assert len(df) == 2019

    output_path = monai_path / "predicted_data" / "float16"
    output_path.mkdir(exist_ok=True, parents=True)

    cfg.dataset.use_segmentations = True

    # import pytorch_lightning
    # pytorch_lightning.seed_everything(cfg.model.seed)
    for fold, ckpt_path in ckpt_dict.items():
        print(f"* fold {fold}")

        target_dir = output_path / f"fold{fold}"
        # if target_dir.exists():
        #     print(f"[Info] Skipped {target_dir}")
        #     continue

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

        df["npz_predicted_segmentations_path"] = df["StudyInstanceUID"].map(lambda uid: target_dir / f"{uid}.npz")

        df["is_predicted"] = df["npz_predicted_segmentations_path"].map(lambda p: p.exists())
        n_exists = np.count_nonzero(df["is_predicted"])
        if n_exists > 0:
            print(f"{len(df)} -> {len(df) - n_exists}")
            df = df[~df["is_predicted"]]

        batch = 50
        for i in tqdm.trange(int(np.ceil(len(df) / batch)), desc="predict"):
            target_df = df.iloc[batch * i: batch * (i + 1)]
            datamodule = CSFD.monai.CSFDDataModule(cfg, target_df)
            predicted = tl.predict(module, datamodule)
            with torch.no_grad():
                predicted = torch.concat(predicted).float().sigmoid().numpy()
            predicted = predicted.astype(np.float16)

            assert len(predicted) == len(target_df)
            for data, p in zip(tqdm.tqdm(predicted, desc="save"), target_df["npz_predicted_segmentations_path"]):
                np.savez_compressed(p, data)
