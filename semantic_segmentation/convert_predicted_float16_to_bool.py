import pathlib
import numpy as np
import tqdm

predicted_data_dir_path = pathlib.Path("predicted_data3")

src_dir_path = predicted_data_dir_path / "float16"
dst_dir_path = predicted_data_dir_path / "uint8"
dst_dir_path.mkdir(exist_ok=True)

for fold_dir in src_dir_path.glob("*"):
    print(f"* {fold_dir}")
    for src_path in tqdm.tqdm(list(fold_dir.glob("*.npz"))):
        dst_path = dst_dir_path / src_path.relative_to(src_dir_path)
        if dst_path.exists():
            continue
        segmentations = np.load(src_path)["arr_0"]
        dst_path.parent.mkdir(exist_ok=True)
        np.savez_compressed(dst_path, segmentations.astype(np.uint8))

