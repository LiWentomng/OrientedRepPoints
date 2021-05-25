from .dota import DotaDataset
from .registry import DATASETS


@DATASETS.register_module
class UCASAODDataset(DotaDataset):
    CLASSES = ('car', 'airplane',)