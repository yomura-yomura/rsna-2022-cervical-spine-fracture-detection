import CSFD.monai.from_checkpoint
import torch
import CSFD.data.io.three_dimensions


cfg, ckpt_dict = CSFD.monai.from_checkpoint.load_cfg_and_checkpoints("../monai/models/EfficientNetBN_folds4_test-v4.1")
df = CSFD.data.io.three_dimensions.get_df(cfg.dataset)

module = CSFD.monai.CSFDModule.load_from_checkpoint(
    str(ckpt_dict[0]), cfg=cfg, map_location=torch.device("cuda")
).cuda().half()


for batch, predicted_list in CSFD.monai.from_checkpoint.predict_on_datamodule_wide(cfg, ckpt_dict, df):
    break

torch.cuda.empty_cache()
batch["data"] = batch["data"].cuda()
module(batch)
module.model.forward(batch["data"])
