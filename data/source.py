from rainbowneko.data.source import WebDatasetSource, ImageLabelSource
import numpy as np
import io
import json
from PIL import Image
from tqdm import tqdm
import imageio.v3 as iio
from typing import Dict, Any


class ZerochanWebDatasetSource(WebDatasetSource):
    def __iter__(self):
        self.pipeline_iter = iter(self.pipeline)
        self.cls_id_map = {}
        return self

    def __next__(self):

        while True:
            try:
                data = next(self.pipeline_iter)
                image = iio.imread(io.BytesIO(data['webp']), extension=".webp")
                image = Image.fromarray(image)
                # with Image.open(io.BytesIO(data['webp'])) as image:
                #     image = image.convert('RGB')
                break
            except (OSError, ValueError) as e:
                print(f"[Warning] Failed to load data: {e}")
                import traceback
                traceback.print_exc()
        
        img_id = data['__key__']
        labels = json.loads(data['json'].decode('utf-8'))

        cls_name = labels['character']
        if cls_name not in self.cls_id_map:
            self.cls_id_map[cls_name] = len(self.cls_id_map)

        return {
            'id': img_id,
            'image': image,
            'label': self.cls_id_map[cls_name],
        }

class StyleWebDatasetSource(WebDatasetSource):
    def __iter__(self):
        self.pipeline_iter = iter(self.pipeline)
        self.cls_id_map = {}
        self.pbar = tqdm(total=self.size, desc="Loading data")
        return self

    def __next__(self):

        while True:
            try:
                data = next(self.pipeline_iter)
                self.pbar.update(1)

                img_id = data['__key__']
                labels = json.loads(data['json'].decode('utf-8'))

                if len(labels['artist_tags']) == 0:
                    continue

                cls_name = labels['artist_tags'][0]
                if cls_name not in self.cls_id_map:
                    self.cls_id_map[cls_name] = len(self.cls_id_map)

                with Image.open(io.BytesIO(data['webp'])) as image:
                    image = image.convert('RGB')
                break
            except (OSError, ValueError) as e:
                print(f"[Warning] Failed to load data: {e}")
                import traceback
                traceback.print_exc()
        
        
        if cls_name not in self.cls_id_map:
            self.cls_id_map[cls_name] = len(self.cls_id_map)

        return {
            'id': img_id,
            'image': image,
            'label': self.cls_id_map[cls_name],
        }

class StyleImageLabelSource(ImageLabelSource):
    def __init__(self, img_root, label_file, **kwargs):
        super().__init__(img_root, label_file, **kwargs)
        self.cls_id_map = {}

    def __getitem__(self, index) -> Dict[str, Any]:
        img_id = self.img_ids[index]
        path = self.img_root / img_id
        cls_name = self.label_dict.get(self.img_ids[index], None)
        if cls_name not in self.cls_id_map:
            self.cls_id_map[cls_name] = len(self.cls_id_map)

        return {
            'id': img_id,
            'image': path,
            'label': self.cls_id_map[cls_name],
        }