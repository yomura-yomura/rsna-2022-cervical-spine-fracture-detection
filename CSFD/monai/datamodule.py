import logging
from pytorch_lightning import LightningDataModule
import os
import CSFD.data
import CSFD.data.io.three_dimensions
import CSFD.data.io_with_cfg.three_dimensions
import CSFD.monai.transforms
from monai.data import DataLoader
from .dataset import Cropped3DDataset, Cropped2DDataset, CacheDataset


__all__ = [
    "CSFDDataModule",
    "CSFDSemanticSegmentationDataModule",
    "CSFDCropped3DDataModule", "CSFDCropped2DDataModule"
]


class CSFDDataModule(LightningDataModule):
    def __init__(self, cfg, df=None):
        super().__init__()

        self.cfg = cfg

        if df is None:
            self.df = CSFD.data.io_with_cfg.three_dimensions.get_df(self.cfg.dataset)
        else:
            self.df = df

        # other configs
        self.num_workers = (
            self.cfg.dataset.num_workers
            if self.cfg.dataset.num_workers is not None
            else os.cpu_count()
        )

        # need to be defined in setup()
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        # self.label_names = {}
        # self.train_transforms = train_transforms
        # self.valid_transforms = valid_transforms

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = CacheDataset(
                self.df[self.df["fold"] != self.cfg.dataset.cv.fold].to_dict("records"),
                CSFD.monai.transforms.get_transforms(self.cfg, is_train=True),
                cache_rate=self.cfg.dataset.train_cache_rate
            )
            logging.info(f"training dataset: {len(self.train_dataset)}")
        if stage in ("fit", "validate"):
            self.valid_dataset = CacheDataset(
                self.df[self.df["fold"] == self.cfg.dataset.cv.fold].to_dict("records"),
                CSFD.monai.transforms.get_transforms(self.cfg, is_train=False),
                cache_rate=self.cfg.dataset.valid_cache_rate
            )
            logging.info(f"validation dataset: {len(self.valid_dataset)}")
        if stage == "predict":
            self.test_dataset = CacheDataset(
                self.df.to_dict("records"),
                CSFD.monai.transforms.get_transforms(self.cfg, is_train=False),
                cache_rate=0
            )
            logging.info(f"test dataset: {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.cfg.dataset.train_batch_size,
            num_workers=self.num_workers,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            shuffle=False,
            batch_size=self.cfg.dataset.valid_batch_size,
            num_workers=self.num_workers,
            drop_last=True
        )

    def predict_dataloader(self):
        if not self.test_dataset:
            logging.warning('no testing data found')
            return
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.cfg.dataset.test_batch_size,
            num_workers=self.num_workers
        )


class CSFDSemanticSegmentationDataModule(CSFDDataModule):
    def __init__(self, cfg, df=None):
        super(CSFDSemanticSegmentationDataModule, self).__init__(cfg, df)
        assert cfg.dataset.use_segmentation is True

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = CacheDataset(
                self.df[self.df["fold"] != self.cfg.dataset.cv.fold].to_dict("records"),
                CSFD.monai.transforms.get_transforms(self.cfg, is_train=True, use_segmentation=True),
                cache_rate=self.cfg.dataset.train_cache_rate
            )
            logging.info(f"training dataset: {len(self.train_dataset)}")
        if stage in ("fit", "validate"):
            self.valid_dataset = CacheDataset(
                self.df[self.df["fold"] == self.cfg.dataset.cv.fold].to_dict("records"),
                CSFD.monai.transforms.get_transforms(self.cfg, is_train=False, use_segmentation=True),
                cache_rate=self.cfg.dataset.valid_cache_rate
            )
            logging.info(f"validation dataset: {len(self.valid_dataset)}")
        if stage == "predict":
            self.test_dataset = CacheDataset(
                self.df.to_dict("records"),
                CSFD.monai.transforms.get_transforms(self.cfg, is_train=False, use_segmentation=True),
                cache_rate=0
            )
            logging.info(f"test dataset: {len(self.test_dataset)}")


class CSFDCropped3DDataModule(CSFDDataModule):
    def setup(self, stage=None):
        super(CSFDCropped3DDataModule, self).setup(stage)

        assert self.cfg.dataset.semantic_segmentation_bb_path is not None

        if self.train_dataset is not None:
            self.train_dataset = Cropped3DDataset(self.cfg, self.train_dataset)
        if self.valid_dataset is not None:
            self.valid_dataset = Cropped3DDataset(self.cfg, self.valid_dataset)
        if self.test_dataset is not None:
            self.test_dataset = Cropped3DDataset(self.cfg, self.test_dataset)


class CSFDCropped2DDataModule(CSFDDataModule):
    def setup(self, stage=None):
        super(CSFDCropped2DDataModule, self).setup(stage)

        assert self.cfg.dataset.semantic_segmentation_bb_path is not None

        if self.train_dataset is not None:
            self.train_dataset = Cropped2DDataset(self.cfg, self.train_dataset)
        if self.valid_dataset is not None:
            self.valid_dataset = Cropped2DDataset(self.cfg, self.valid_dataset)
        if self.test_dataset is not None:
            self.test_dataset = Cropped2DDataset(self.cfg, self.test_dataset)
