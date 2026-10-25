"""评测集用的最小 tar 工具: 扫描成员的数据偏移 (只读 512B 头)、按偏移读字节、写打包 tar 并返回偏移.
(训练用的完整 tar 索引 / map-style 数据集在 v3/data, 这里只服务评测集, 不依赖它)"""
from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path


def _parse_size(field: bytes) -> int:
    if field[0] & 0x80:  # base-256 (GNU) 大文件
        return int.from_bytes(field[1:], 'big')
    s = field.rstrip(b'\0 ').strip()
    return int(s, 8) if s else 0


def scan_tar(path) -> list:
    """返回 [(member_name, data_offset, size)], 支持 ustar 前缀和 GNU longname ('L'), 跳过 pax 头"""
    out = []
    fd = os.open(str(path), os.O_RDONLY)
    try:
        pos, end = 0, os.fstat(fd).st_size
        longname = None
        while pos + 512 <= end:
            hdr = os.pread(fd, 512, pos)
            if hdr == b'\0' * 512:
                break
            name = hdr[0:100].split(b'\0', 1)[0]
            prefix = hdr[345:500].split(b'\0', 1)[0]
            size = _parse_size(hdr[124:136])
            typ = hdr[156:157]
            data = pos + 512
            if typ == b'L':
                longname = os.pread(fd, size, data).split(b'\0', 1)[0]
            elif typ in (b'x', b'g'):
                pass
            elif typ in (b'0', b'\0', b'7'):
                full = longname or ((prefix + b'/' + name) if prefix else name)
                out.append((full.decode('utf-8', 'surrogateescape'), data, size))
                longname = None
            pos = data + (size + 511) // 512 * 512
    finally:
        os.close(fd)
    return out


def read_at(path, offset: int, size: int) -> bytes:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        return os.pread(fd, size, offset)
    finally:
        os.close(fd)


class PackWriter:
    """按大小切分的 tar 写入器; write() 返回 (tar 文件名, 数据偏移, 大小)"""

    def __init__(self, out_dir, prefix: str, max_bytes: int = 1 << 30):
        self.out_dir, self.prefix, self.max_bytes = Path(out_dir), prefix, max_bytes
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.idx, self.tf, self.name = -1, None, None

    def _roll(self):
        if self.tf:
            self.tf.close()
        self.idx += 1
        self.name = f'{self.prefix}-{self.idx:05d}.tar'
        self.tf = tarfile.open(self.out_dir / self.name, 'w', format=tarfile.USTAR_FORMAT)

    def write(self, member: str, data: bytes):
        if self.tf is None or self.tf.offset >= self.max_bytes:
            self._roll()
        ti = tarfile.TarInfo(member)
        ti.size = len(data)
        ti.mtime = 0
        assert len(member.encode()) <= 100, member  # 单个 512B 头, 偏移 = 头位置 + 512
        offset = self.tf.offset + tarfile.BLOCKSIZE
        self.tf.addfile(ti, io.BytesIO(data))
        return self.name, offset, len(data)

    def close(self):
        if self.tf:
            self.tf.close()
            self.tf = None
