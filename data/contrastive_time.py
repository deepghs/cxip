from rainbowneko.data import ImageLabelSource
from rainbowneko.data.source import DataSource
from rainbowneko.data.bucket import BaseBucket
from typing import Dict

class TimeContrastDataset(ImageLabelDataset):
    def __init__(self, bucket: BaseBucket = None, source: Dict[str, DataSource] = None, time_decay=50000000, **kwargs):
        super().__init__(bucket, source, **kwargs)

    def load_label(self, img_path: str, data_source: DataSource):
        label = data_source.load_label(img_path)
        label['time'] = int(img_path.rsplit('.', 1)[0].rsplit('_', 1)[1])
        return {'label': label}

