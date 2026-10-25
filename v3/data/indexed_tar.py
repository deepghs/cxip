"""Map-style random-access dataset over WebDataset tar shards via a parquet offset index.

The index (see ``v3/data/build_index.py``) stores, per sample, the tar shard (relative to the
dataset root), the byte offset of the image member's data and its size. Reads are a single
``os.pread`` on a per-process cached file descriptor: descriptors are opened lazily and
re-opened after ``fork`` (DataLoader workers), so the dataset is safe to construct in the main
process and hand to a multi-worker DataLoader.

Metadata columns are kept as Arrow arrays (no per-row Python objects in the parent process), so
forked workers do not slowly copy the parent's heap through refcount writes.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

try:
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - torch is optional for index tooling
    Dataset = object  # type: ignore[misc,assignment]

Image.MAX_IMAGE_PIXELS = None
_CORE = ('sample_key', 'shard', 'offset', 'size', 'ext', 'label')


def to_rgb_white(im: Image.Image) -> Image.Image:
    """RGB image; any transparency is alpha-composited onto a white background."""
    if im.mode == 'RGB':
        return im
    has_alpha = im.mode in ('RGBA', 'LA', 'PA', 'RGBa', 'La') or (
        im.mode == 'P' and 'transparency' in im.info)
    if has_alpha:
        im = im.convert('RGBA')
        bg = Image.new('RGBA', im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        return bg.convert('RGB')
    return im.convert('RGB')


def decode_image(data: bytes, jpeg_draft: int | None = None) -> Image.Image:
    """Decode to RGB (white background). ``jpeg_draft``: let libjpeg decode JPEGs at a reduced
    DCT scale (1/2, 1/4, 1/8) while keeping both sides >= jpeg_draft (much faster for small
    training resolutions; other formats are unaffected)."""
    with Image.open(io.BytesIO(data)) as im:
        if jpeg_draft and im.format == 'JPEG':
            im.draft('RGB', (jpeg_draft, jpeg_draft))
        im.load()
        return to_rgb_white(im)


def _load_table(index, columns: Sequence[str] | None) -> pa.Table:
    if isinstance(index, pa.Table):
        table = index
    elif isinstance(index, pd.DataFrame):
        table = pa.Table.from_pandas(index, preserve_index=False)
    else:
        want = None if columns is None else list(dict.fromkeys([*_CORE, *columns]))
        if want is not None:
            available = set(pq.read_schema(index).names)
            missing = [c for c in want if c not in available]
            if missing:
                raise KeyError(f'columns not in index {index}: {missing}')
        table = pq.read_table(index, columns=want)
    return table


class IndexedTarDataset(Dataset):
    """``ds[i] -> dict(image=PIL RGB, label=int, index=i, **metadata)``.

    Args:
        root: dataset root; index ``shard`` paths are relative to it.
        index: parquet path, pandas DataFrame or pyarrow Table with the index columns.
        transform: optional callable applied to the PIL image (result stored under ``image``).
        columns: metadata columns to return per sample (default: none). ``'all'`` returns every
            non-core column of the index.
        return_bytes: return the raw encoded bytes under ``image`` instead of a decoded image.
        jpeg_draft: see :func:`decode_image` (e.g. the training resolution); None = full decode.
    """

    def __init__(self, root: str | os.PathLike, index, transform: Callable | None = None,
                 columns: Sequence[str] | str | None = None, return_bytes: bool = False,
                 jpeg_draft: int | None = None):
        self.root = Path(root)
        cols = None if columns == 'all' else list(columns or [])
        table = _load_table(index, cols)
        if columns == 'all':
            cols = [c for c in table.column_names if c not in _CORE]
        self.columns: list[str] = list(cols)
        shard = table.column('shard').to_pandas()
        codes, uniques = pd.factorize(shard, sort=True)
        self.shards: list[str] = [str(u) for u in uniques]
        self.shard_idx = codes.astype(np.int32)
        self.offsets = table.column('offset').to_numpy().astype(np.int64)
        self.sizes = table.column('size').to_numpy().astype(np.int64)
        self.labels = table.column('label').to_numpy().astype(np.int64)
        self.keys = table.column('sample_key').combine_chunks()
        self._meta = {c: table.column(c).combine_chunks() for c in self.columns}
        self.transform = transform
        self.return_bytes = return_bytes
        self.jpeg_draft = jpeg_draft
        self._fds: dict[int, int] = {}
        self._pid = os.getpid()

    # -- file descriptors ------------------------------------------------------------------
    def _fd(self, s: int) -> int:
        if self._pid != os.getpid():  # forked: never share descriptors / offsets with the parent
            self._fds = {}
            self._pid = os.getpid()
        fd = self._fds.get(s)
        if fd is None:
            fd = os.open(self.root / self.shards[s], os.O_RDONLY)
            self._fds[s] = fd
        return fd

    def close(self):
        if self._pid == os.getpid():
            for fd in self._fds.values():
                try:
                    os.close(fd)
                except OSError:
                    pass
        self._fds = {}

    def __del__(self):  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_fds'] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._pid = os.getpid()

    # -- access ----------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.offsets)

    def read_bytes(self, i: int) -> bytes:
        size = int(self.sizes[i])
        data = os.pread(self._fd(int(self.shard_idx[i])), size, int(self.offsets[i]))
        if len(data) != size:
            raise IOError(f'short read for sample {i} ({self.keys[i]}): {len(data)}/{size}')
        return data

    def metadata(self, i: int) -> dict:
        return {c: arr[i].as_py() for c, arr in self._meta.items()}

    def __getitem__(self, i: int) -> dict:
        i = int(i)
        data = self.read_bytes(i)
        if self.return_bytes:
            image = data
        else:
            image = decode_image(data, self.jpeg_draft)
            if self.transform is not None:
                image = self.transform(image)
        out = {'image': image, 'label': int(self.labels[i]), 'index': i}
        for c, arr in self._meta.items():
            out[c] = arr[i].as_py()
        return out

    def column(self, name: str) -> np.ndarray:
        """Whole column as numpy: ``label`` or one of the selected metadata ``columns``."""
        if name == 'label':
            return self.labels
        if name in self._meta:
            return self._meta[name].to_numpy(zero_copy_only=False)
        raise KeyError(name)


def list_collate(batch: Iterable[dict]) -> dict:
    """Collate dicts of PIL images / mixed metadata into dict of lists (labels as a tensor)."""
    import torch
    batch = list(batch)
    out = {k: [b[k] for b in batch] for k in batch[0]}
    out['label'] = torch.as_tensor(out['label'], dtype=torch.long)
    out['index'] = torch.as_tensor(out['index'], dtype=torch.long)
    if all(hasattr(x, 'shape') for x in out['image']):
        out['image'] = torch.stack(out['image'])
    return out
