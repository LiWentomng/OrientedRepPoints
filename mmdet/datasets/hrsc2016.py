from .dota import DotaDataset
from .registry import DATASETS


@DATASETS.register_module
class HRSC2016Dataset(DotaDataset):
    CLASSES = ('ship',)