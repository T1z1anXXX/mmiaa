from vilt.datasets import AVADataset
from .datamodule_base import BaseDataModule


class AVADataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return AVADataset

    @property
    def dataset_name(self):
        return "ava"

    def setup(self, stage):
        super().setup(stage)
