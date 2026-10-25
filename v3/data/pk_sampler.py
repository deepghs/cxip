"""Distributed P x K batch sampler for metric learning (CCIP / CSIP).

Every *global* batch contains P distinct classes x K samples per class:

- class sampling probability is proportional to ``n_c ** alpha`` (alpha=0: class-balanced,
  alpha=1: instance-balanced), without replacement inside a batch;
- a fraction ``group_frac`` of the P classes is drawn as whole groups of classes sharing a
  ``group_ids`` value (e.g. same series/copyright) -> same-franchise hard negatives;
- a fraction ``cross_frac`` of the P classes is drawn as "crossed" groups: classes that share a
  sample-level content value (``cross_ids``, e.g. the character of a single-character danbooru
  post for CSIP artists) each contribute one sample with that shared content -> same content,
  different class hard negatives;
- the K samples of a class are distinct when the class has >= K samples (otherwise all samples
  first, then repeats), and with ``avoid_same`` (e.g. character id for CSIP, artist id for CCIP)
  the K samples prefer distinct ``avoid_same`` values;
- with ``spread`` (a per-sample number, e.g. the post year for CSIP) the K samples are taken one per
  quantile stratum of the class's candidate pool -> positives span the class's range (style drift).

Distribution: all ranks build the identical global batch from ``(seed, epoch, batch_idx)`` and
rank r takes classes ``[r*P/W, (r+1)*P/W)`` of it, i.e. **classes are kept whole within a rank**
(all K samples of a class are on the same rank), so per-rank losses (ArcFace, local SigLIP) see
positives; group/crossed classes are laid out contiguously so they mostly land on one rank too.
Losses that all-gather (SigLIP) see the whole global batch. P must be divisible by world_size.

Batches are deterministic per (seed, epoch, batch index), independent of world size except
for the slicing; ``start_batch`` allows exact resume inside an epoch.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterator, Sequence

import numpy as np

try:
    from torch.utils.data import Sampler
except ImportError:  # pragma: no cover
    Sampler = object  # type: ignore[misc,assignment]


def _csr(keys: np.ndarray, n_keys: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(keys, kind='stable')
    counts = np.bincount(keys, minlength=n_keys)
    offsets = np.zeros(n_keys + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    return order, offsets


def _class_group(group_ids: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Per-class group id (majority over the class's samples, ignoring < 0); -1 = no group."""
    out = np.full(n_classes, -1, dtype=np.int64)
    ok = group_ids >= 0
    if not ok.any():
        return out
    pairs = np.stack([labels[ok], group_ids[ok]], 1)
    uniq, cnt = np.unique(pairs, axis=0, return_counts=True)
    order = np.lexsort((-cnt, uniq[:, 0]))  # by class, then most frequent first
    uniq = uniq[order]
    first = np.r_[True, uniq[1:, 0] != uniq[:-1, 0]]
    out[uniq[first, 0]] = uniq[first, 1]
    return out


class PKBatchSampler(Sampler):
    """Yields lists of dataset indices (this rank's share of each global P x K batch).

    Args:
        labels: per-sample class labels (non-negative ints; need not be contiguous).
        P, K: classes per global batch, samples per class.
        alpha: class sampling exponent, p_c ~ n_c ** alpha.
        group_ids: per-sample (len == len(labels)) or per-class (len == max(label)+1) group id;
            values < 0 mean "no group". Per-sample ids are reduced to the class majority.
        group_frac: fraction of P drawn as whole groups.
        group_size: max classes taken from one group per draw (default 4).
        cross_ids: per-sample content id for crossed groups (< 0 = none).
        cross_frac: fraction of P drawn as crossed groups; cross_size: classes per crossed group.
        avoid_same: per-sample values the K samples of a class should not share (< 0 = unknown).
        spread: per-sample numbers the K samples should spread over (< 0 = unknown).
        rank, world_size: distributed slice; P % world_size == 0.
        seed: base seed; call ``set_epoch`` every epoch.
        num_batches: batches per epoch (default len(labels) // (P*K)).
        min_count: classes with fewer samples are never sampled (default 1).
        start_batch: skip the first batches of the current epoch (resume).
    """

    def __init__(self, labels: Sequence[int], P: int, K: int, *, alpha: float = 0.5,
                 group_ids: Sequence[int] | None = None, group_frac: float = 0.25, group_size: int = 4,
                 cross_ids: Sequence[int] | None = None, cross_frac: float = 0.0, cross_size: int = 4,
                 rank: int = 0, world_size: int = 1, seed: int = 0, num_batches: int | None = None,
                 avoid_same: Sequence[int] | None = None, min_count: int = 1, start_batch: int = 0,
                 spread: Sequence[int] | None = None):
        labels = np.asarray(labels, dtype=np.int64)
        if labels.ndim != 1 or (labels < 0).any():
            raise ValueError('labels must be a 1-d array of non-negative ints')
        if P % world_size:
            raise ValueError(f'P={P} must be divisible by world_size={world_size}')
        if not 0 <= rank < world_size:
            raise ValueError('bad rank')
        self.labels, self.P, self.K = labels, int(P), int(K)
        self.alpha, self.rank, self.world_size = float(alpha), int(rank), int(world_size)
        self.seed, self.epoch, self.start_batch = int(seed), 0, int(start_batch)
        self.group_frac, self.group_size = float(group_frac), max(int(group_size), 2)
        self.cross_frac, self.cross_size = float(cross_frac), max(int(cross_size), 2)
        n_classes = int(labels.max()) + 1
        self.order, self.offsets = _csr(labels, n_classes)
        counts = np.diff(self.offsets)
        self.counts = counts
        self.valid = np.nonzero(counts >= max(min_count, 1))[0]
        if len(self.valid) < self.P:
            raise ValueError(f'only {len(self.valid)} classes with >= {min_count} samples, need P={P}')
        w = np.zeros(n_classes)
        w[self.valid] = counts[self.valid].astype(np.float64) ** self.alpha
        self.class_w = w
        self.class_p = w / w.sum()
        self.num_batches = int(num_batches) if num_batches else max(len(labels) // (self.P * self.K), 1)
        self.avoid = None if avoid_same is None else np.asarray(avoid_same, dtype=np.int64)
        if self.avoid is not None and len(self.avoid) != len(labels):
            raise ValueError('avoid_same must align with labels')
        self.spread = None if spread is None else np.asarray(spread, dtype=np.int64)
        if self.spread is not None and len(self.spread) != len(labels):
            raise ValueError('spread must align with labels')

        # ---- groups of classes
        self.group_members: list[np.ndarray] = []
        self.group_p = None
        if group_ids is not None and self.group_frac > 0:
            g = np.asarray(group_ids, dtype=np.int64)
            if len(g) == len(labels):
                cg = _class_group(g, labels, n_classes)
            elif len(g) >= n_classes:
                cg = g[:n_classes].copy()
            else:
                raise ValueError('group_ids must be per-sample or per-class')
            cg[w == 0] = -1
            members = defaultdict(list)
            for c in np.nonzero(cg >= 0)[0]:
                members[int(cg[c])].append(int(c))
            self.group_members = [np.asarray(m) for m in members.values() if len(m) >= 2]
            if self.group_members:
                gw = np.array([w[m].sum() for m in self.group_members])
                self.group_p = gw / gw.sum()

        # ---- crossed groups: content value -> {class -> sample indices}
        self.cross_values: list[tuple[np.ndarray, list[np.ndarray]]] = []
        self.cross_p = None
        if cross_ids is not None and self.cross_frac > 0:
            self.cross_values = build_crossed_index(cross_ids, labels, valid_classes=self.valid)
            if self.cross_values:
                cw = np.array([len(cls) for cls, _ in self.cross_values], dtype=np.float64)
                self.cross_p = cw / cw.sum()

    # ------------------------------------------------------------------------------------
    def set_epoch(self, epoch: int, start_batch: int = 0):
        self.epoch, self.start_batch = int(epoch), int(start_batch)

    def __len__(self) -> int:
        return self.num_batches - self.start_batch

    def __iter__(self) -> Iterator[list[int]]:
        per_rank = self.P // self.world_size
        lo, hi = self.rank * per_rank, (self.rank + 1) * per_rank
        for b in range(self.start_batch, self.num_batches):
            classes, fixed = self.global_batch_classes(b)
            rng = np.random.default_rng([self.seed, self.epoch, b, 1])
            # consume the rng identically on all ranks -> identical global batch
            out = []
            for j, c in enumerate(classes):
                idx = self._sample_class(rng, int(c), fixed.get(j))
                if lo <= j < hi:
                    out.extend(idx)
            yield out

    def global_batch(self, b: int, epoch: int | None = None) -> list[list[int]]:
        """Full global batch as P lists of K indices (for tests / inspection)."""
        if epoch is not None:
            self.epoch = epoch
        classes, fixed = self.global_batch_classes(b)
        rng = np.random.default_rng([self.seed, self.epoch, b, 1])
        return [self._sample_class(rng, int(c), fixed.get(j)) for j, c in enumerate(classes)]

    def global_batch_classes(self, b: int) -> tuple[np.ndarray, dict[int, int]]:
        """P distinct classes (group/crossed units contiguous) + {position: forced sample index}."""
        rng = np.random.default_rng([self.seed, self.epoch, b, 0])
        P = self.P
        chosen: list[int] = []
        taken = np.zeros(len(self.class_w), dtype=bool)
        units: list[list[tuple[int, int | None]]] = []

        n_cross = int(round(self.cross_frac * P)) if self.cross_p is not None else 0
        tries = 0
        while n_cross - sum(len(u) for u in units) >= 2 and tries < 50:
            tries += 1
            v = rng.choice(len(self.cross_values), p=self.cross_p)
            cls, samples = self.cross_values[v]
            free = np.nonzero(~taken[cls])[0]
            if len(free) < 2:
                continue
            room = n_cross - sum(len(u) for u in units)
            k = min(self.cross_size, len(free), room)
            pw = self.class_w[cls[free]]
            pick = rng.choice(free, size=k, replace=False, p=pw / pw.sum())
            unit = []
            for f in pick:
                c = int(cls[f])
                taken[c] = True
                unit.append((c, int(rng.choice(samples[f]))))
            units.append(unit)

        n_group = int(round(self.group_frac * P)) if self.group_p is not None else 0
        n_group = min(n_group, P - sum(len(u) for u in units))
        got, tries = 0, 0
        while n_group - got >= 2 and tries < 50:
            tries += 1
            m = self.group_members[rng.choice(len(self.group_members), p=self.group_p)]
            free = m[~taken[m]]
            if len(free) < 2:
                continue
            k = min(self.group_size, len(free), n_group - got)
            pw = self.class_w[free]
            pick = rng.choice(free, size=k, replace=False, p=pw / pw.sum())
            taken[pick] = True
            units.append([(int(c), None) for c in pick])
            got += k

        n_rest = P - sum(len(u) for u in units)
        if n_rest > 0:
            p = self.class_p.copy()
            p[taken] = 0
            p /= p.sum()
            rest = rng.choice(len(p), size=n_rest, replace=False, p=p)
            units.extend([[(int(c), None)] for c in rest])
        perm = rng.permutation(len(units))
        fixed: dict[int, int] = {}
        for u in perm:
            for c, forced in units[u]:
                if forced is not None:
                    fixed[len(chosen)] = forced
                chosen.append(c)
        return np.asarray(chosen, dtype=np.int64), fixed

    def _sample_class(self, rng: np.random.Generator, c: int, forced: int | None = None) -> list[int]:
        K = self.K
        members = self.order[self.offsets[c]:self.offsets[c + 1]]
        n = len(members)
        if n > 16 * K:  # cheap candidate pool for big classes
            cand = members[rng.choice(n, size=16 * K, replace=False)]
        else:
            cand = members[rng.permutation(n)]
        if forced is not None:
            cand = np.concatenate([[forced], cand[cand != forced]])
        if self.avoid is not None and len(cand) > 1:
            vals = self.avoid[cand]
            seen: set[int] = set()
            first, rest = [], []
            for i, v in zip(cand.tolist(), vals.tolist()):
                if v >= 0 and v in seen:
                    rest.append(i)
                else:
                    if v >= 0:
                        seen.add(v)
                    first.append(i)
            cand = np.asarray(first + rest, dtype=np.int64)
        if self.spread is not None and len(cand) > K:
            cand = self._spread_pick(cand, K, forced is not None)
        if len(cand) >= K:
            return cand[:K].tolist()
        reps = cand[rng.permutation(len(cand))] if len(cand) > 1 else cand
        out = cand.tolist()
        while len(out) < K:
            out.extend(reps[:K - len(out)].tolist())
        return out

    def _spread_pick(self, cand: np.ndarray, K: int, keep_first: bool) -> np.ndarray:
        """Reorder ``cand`` so its first K entries cover the quantile strata of ``spread`` (one per
        stratum; within a stratum the earliest candidate in the current order, i.e. avoid-preferred)."""
        head = cand[:1] if keep_first else cand[:0]
        pool = cand[len(head):]
        v = self.spread[pool]
        known = np.nonzero(v >= 0)[0]
        k = K - len(head)
        if k <= 0 or len(known) < k:
            return cand
        by_val = known[np.argsort(v[known], kind='stable')]
        picks = [int(st.min()) for st in np.array_split(by_val, k)]  # min position = earliest in avoid order
        mask = np.zeros(len(pool), dtype=bool)
        mask[picks] = True
        return np.concatenate([head, pool[sorted(picks)], pool[~mask]])


def build_crossed_index(cross_ids: Sequence[int], labels: Sequence[int], *, min_classes: int = 2,
                        valid_classes: np.ndarray | None = None) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    """Content value -> (classes having it, per-class sample indices with it).

    Only values present in >= ``min_classes`` distinct classes are kept. For CSIP: cross_ids =
    character id of single-character posts (else -1), labels = artist ids.
    """
    x = np.asarray(cross_ids, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int64)
    keep = x >= 0
    if valid_classes is not None:
        ok = np.zeros(int(y.max()) + 1, dtype=bool)
        ok[valid_classes] = True
        keep &= ok[y]
    idx = np.nonzero(keep)[0]
    if not len(idx):
        return []
    order = np.lexsort((y[idx], x[idx]))
    idx = idx[order]
    xv, yv = x[idx], y[idx]
    out = []
    bounds = np.nonzero(np.r_[True, xv[1:] != xv[:-1], True])[0]
    for s, e in zip(bounds[:-1], bounds[1:]):
        ys = yv[s:e]
        cb = np.nonzero(np.r_[True, ys[1:] != ys[:-1], True])[0]
        if len(cb) - 1 < min_classes:
            continue
        classes = ys[cb[:-1]]
        samples = [idx[s + a:s + b] for a, b in zip(cb[:-1], cb[1:])]
        out.append((classes, samples))
    return out


def crossed_batches(cross_ids: Sequence[int], labels: Sequence[int], n_groups: int, group_size: int = 4,
                    seed: int = 0, crossed_index=None) -> list[list[int]]:
    """Standalone helper: ``n_groups`` groups of sample indices sharing a content value (e.g. the
    same character) with pairwise different labels (artists). Content values are drawn
    proportional to the number of classes that have them."""
    index = crossed_index if crossed_index is not None else build_crossed_index(cross_ids, labels)
    if not index:
        return []
    rng = np.random.default_rng(seed)
    w = np.array([len(c) for c, _ in index], dtype=np.float64)
    out = []
    for v in rng.choice(len(index), size=n_groups, p=w / w.sum()):
        classes, samples = index[v]
        pick = rng.choice(len(classes), size=min(group_size, len(classes)), replace=False)
        out.append([int(rng.choice(samples[p])) for p in pick])
    return out


def single_value_ids(lists: Sequence) -> np.ndarray:
    """Per-sample id of the only element of a list column (e.g. ``characters``); -1 if the list
    is empty or has several elements. Useful as ``cross_ids`` / ``avoid_same`` for CSIP."""
    vals = [x[0] if x is not None and len(x) == 1 else None for x in lists]
    return encode_ids(vals)


def encode_ids(values: Sequence, *, missing=(None, '', 'nan')) -> np.ndarray:
    """Map arbitrary hashable values (strings, ...) to int codes; missing -> -1."""
    import pandas as pd
    s = pd.Series(values, dtype=object)
    s = s.where(~s.isin(list(missing)) & s.notna(), None)
    codes, _ = pd.factorize(s, use_na_sentinel=True)
    return codes.astype(np.int64)
