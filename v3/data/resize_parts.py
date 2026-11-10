"""Sequential tar scan -> resize -> large part tar + offset parquet (generic, used by the v4 dataset builders).

Each source tar is one job: its 512-byte headers are walked in file order (no random reads), the wanted members are
decoded and resized (short side <= 640, never upscaled, webp quality 95, alpha kept) and written as ``{key}.webp``
into one part tar per source tar. Next to it ``<part>.parquet`` (key, offset, size, width, height, error) gives the
data offset of every output member, so later steps can pread images without tarfile. Both files are written as
``.tmp`` and renamed; the parquet is renamed last and marks the part as finished (finished parts are skipped).
"""
from __future__ import annotations

import io
import os
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

import pandas as pd
from PIL import Image

PART_COLS = ['key', 'offset', 'size', 'width', 'height', 'error']


def _tar_size(field: bytes) -> int:
    if field[0] & 0x80:                                   # base-256 (GNU, > 8 GiB members)
        return int.from_bytes(field[1:], 'big')
    return int(field.split(b'\0', 1)[0].strip() or b'0', 8)


def _pax_records(buf: bytes) -> dict[str, str]:
    """``"%d key=value\\n"`` records; the length counts the whole record."""
    out, pos = {}, 0
    while pos < len(buf):
        sp = buf.find(b' ', pos)
        if sp < 0:
            break
        try:
            n = int(buf[pos:sp])
        except ValueError:
            break
        if n <= 0:
            break
        rec = buf[sp + 1:pos + n]
        if rec.endswith(b'\n'):
            rec = rec[:-1]
        k, _, v = rec.partition(b'=')
        out[k.decode('utf-8', 'replace')] = v.decode('utf-8', 'replace')
        pos += n
    return out


def iter_tar_members(path: str | os.PathLike) -> Iterator[tuple[str, int, int]]:
    """(name, data_offset, size) of every regular file in tar order, reading only headers (pread).

    Handles GNU long names (``L``), pax headers (``x``: ``path=`` / ``size=`` apply to the next member; ``g`` is
    skipped), the ustar name prefix and base-256 sizes.
    """
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
        off, end = 0, os.fstat(fd).st_size
        long_name, pax = None, {}
        while off + 512 <= end:
            h = os.pread(fd, 512, off)
            if not h.strip(b'\0'):
                break
            size, kind = _tar_size(h[124:136]), h[156:157]
            data_off = off + 512
            if kind == b'L':
                long_name = os.pread(fd, size, data_off).split(b'\0', 1)[0].decode('utf-8', 'replace')
            elif kind == b'x':
                pax = _pax_records(os.pread(fd, size, data_off))
            elif kind in (b'g', b'K'):
                pass
            else:
                if 'size' in pax:
                    size = int(pax['size'])
                if kind in (b'0', b'\0', b'7'):
                    name = pax.get('path') or long_name
                    if name is None:
                        name = h[:100].split(b'\0', 1)[0].decode('utf-8', 'replace')
                        if h[257:262] == b'ustar':
                            prefix = h[345:500].split(b'\0', 1)[0].decode('utf-8', 'replace')
                            if prefix:
                                name = f'{prefix}/{name}'
                    yield name, data_off, size
                long_name, pax = None, {}
            off = data_off + (size + 511) // 512 * 512
    finally:
        os.close(fd)


def _has_alpha(im: Image.Image) -> bool:
    return im.mode in ('RGBA', 'LA', 'PA', 'RGBa', 'La') or 'transparency' in im.info


def resize_image(data: bytes, short_side: int = 640, quality: int = 95) -> tuple[bytes, int, int]:
    """Short side <= ``short_side`` (BICUBIC, never upscaled), webp, RGB or RGBA (alpha kept)."""
    im = Image.open(io.BytesIO(data))
    im.load()
    if im.mode not in ('RGB', 'RGBA'):
        im = im.convert('RGBA' if _has_alpha(im) else 'RGB')
    w, h = im.size
    if min(w, h) > short_side:
        s = short_side / min(w, h)
        im = im.resize((round(w * s), round(h * s)), Image.BICUBIC)
    b = io.BytesIO()
    im.save(b, 'WEBP', quality=quality)
    return b.getvalue(), im.size[0], im.size[1]


def _finished(out_tar: Path) -> bool:
    return out_tar.exists() and out_tar.with_suffix('.parquet').exists()


def process_tar(src: Path, want: dict[str, str], out_tar: Path) -> pd.DataFrame:
    """Resize the ``want`` members (member name -> output key) of ``src`` into ``out_tar`` as ``{key}.webp``.

    Returns one row per wanted member (tar order; members missing from ``src`` last, with an error). Rows with an
    error have no image in the part. Skips (and just reads back) a finished part.
    """
    out_tar = Path(out_tar)
    pq_path = out_tar.with_suffix('.parquet')
    if _finished(out_tar):
        return pd.read_parquet(pq_path)
    out_tar.parent.mkdir(parents=True, exist_ok=True)
    tmp_tar = out_tar.with_name(out_tar.name + '.tmp')
    tmp_pq = pq_path.with_name(pq_path.name + '.tmp')
    rows, seen = [], set()
    fd = os.open(src, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
        with tarfile.open(tmp_tar, 'w', format=tarfile.GNU_FORMAT) as tf:
            for name, off, size in iter_tar_members(src):
                key = want.get(name)
                if key is None or name in seen:
                    continue
                seen.add(name)
                try:
                    img, w, h = resize_image(os.pread(fd, size, off))
                except Exception as e:                     # undecodable / truncated image: record and skip
                    rows.append((key, None, None, None, None, f'{type(e).__name__}: {e}'[:300]))
                    continue
                ti = tarfile.TarInfo(f'{key}.webp')
                ti.size, ti.mode = len(img), 0o644
                tf.addfile(ti, io.BytesIO(img))
                data_off = tf.offset - (len(img) + 511) // 512 * 512
                rows.append((key, data_off, len(img), w, h, None))
    finally:
        os.close(fd)
    for name, key in want.items():
        if name not in seen:
            rows.append((key, None, None, None, None, 'missing in source tar'))
    df = pd.DataFrame(rows, columns=PART_COLS)
    for c in ['offset', 'size', 'width', 'height']:
        df[c] = df[c].astype('Int64')
    df['key'] = df['key'].astype(str)
    df['error'] = df['error'].astype(object)
    df.to_parquet(tmp_pq, index=False)
    os.replace(tmp_tar, out_tar)
    os.replace(tmp_pq, pq_path)                            # parquet last: marks the part finished
    return df


def part_name(src_tar: str | os.PathLike) -> str:
    """Deterministic, collision-free part file name for a source tar (last 4 path components)."""
    return '__'.join(Path(src_tar).parts[-4:])


def _job(args):
    src, want, out_tar = args
    done = _finished(Path(out_tar))
    t0 = time.time()
    df = process_tar(Path(src), want, Path(out_tar))
    return out_tar, df, done, time.time() - t0


def run_jobs(jobs: pd.DataFrame, out_dir: Path, workers: int) -> pd.DataFrame:
    """One process-pool task per source tar (jobs: src_tar, member, key). Returns the merged parts table
    (``part`` = part tar file name relative to ``out_dir``). Source tars that fail as a whole are retried by the
    next run; the call raises after all other tars are done."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(src, dict(zip(g.member, g.key.astype(str))), str(out_dir / part_name(src)))
             for src, g in jobs.groupby('src_tar', sort=True)]
    n_tasks, n_imgs = len(tasks), len(jobs)
    step = max(1, n_tasks // 100)
    print(f'resize: {n_tasks} source tars, {n_imgs} images, workers {workers}', flush=True)
    parts, failed = [], []
    t0, n_done, n_new_img, n_err, n_skipped = time.time(), 0, 0, 0, 0
    with ProcessPoolExecutor(max(1, min(workers, n_tasks))) as ex:
        futs = {ex.submit(_job, t): t[0] for t in tasks}
        for fut in as_completed(futs):
            n_done += 1
            try:
                out_tar, df, skipped, _ = fut.result()
            except Exception as e:
                failed.append((futs[fut], f'{type(e).__name__}: {e}'))
                print(f'FAILED {futs[fut]}: {type(e).__name__}: {e}', flush=True)
                continue
            df = df.assign(part=Path(out_tar).name)
            parts.append(df)
            n_err += int(df.error.notna().sum())
            if skipped:
                n_skipped += 1
            else:
                n_new_img += len(df)
            if n_done % step == 0 or n_done == n_tasks:
                el = time.time() - t0
                print(f'{n_done}/{n_tasks} tars ({n_skipped} already done), {n_new_img} new images, '
                      f'{n_new_img / max(el, 1e-9):.0f} img/s, errors {n_err}, elapsed {el / 60:.1f} min',
                      flush=True)
    if failed:
        raise RuntimeError(f'{len(failed)} source tars failed (rerun to retry): {failed[:5]}')
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=PART_COLS + ['part'])
