"""CCIP 合并数据集 (ccip_merged_nd) 的类别元数据: 作品 (series / franchise) 推断 + Danbooru 角色标签匹配.

用途: 开放集划分 (v3/eval/build_ccip_openset.py) 需要知道每个类属于哪个作品 (按作品分层、整作品剔除),
以及 AnimeDINOv3 预训练时是否把它当作有监督的角色标签见过 (tags.json 中 category=character 的 3,247 个).

作品来源 (按可信度):
  v2 成员: 路径第一级目录 (去掉 `^\\d+_\\d+_` 前缀), 如 Fate_Grand_Order/Lancer_Fate_Grand_Order
  v1 成员: 父目录 (fgo / arknights / azurlane ...; others 无信息)
  zc800k 成员: 名字括号里的限定词, 仅当它是已知作品名时 (如 "Rem (Re:Zero)"; "Caster (Gilgamesh)" 的括号是真名)
  Danbooru: 匹配上的角色标签在 isek-ai/danbooru-tags-2024 里最常共现的 copyright
作品名统一成 norm_key (小写、去重音、只留字母数字), 再用 Danbooru copyright 的蕴含关系 (P(B|A) >= 0.9) 上溯到
franchise 根 (fate/grand order -> fate (series)). 非 Danbooru 写法的作品名 (THE_iDOLM_STER_Cinderella_Girls) 用该作品下
高可信匹配角色的 copyright 投票得到别名.

Danbooru 匹配 (精度优先, 拿不准的记 unknown):
  1. 名字规范化后与 Danbooru 角色标签的基名 (去掉括号限定词) 完全相同, 且作品 franchise 有交集 -> 匹配 (series)
  2. 类没有任何作品信息时, 只接受 >= 2 个词且在 Danbooru 中基名唯一的名字 -> 匹配 (unique_name)
  3. 姓名词序颠倒 (Rin Hoshizora / hoshizora rin) 只在作品相容时接受
  backbone_seen: 匹配到骨干标签 -> seen; 只匹配到非骨干标签 -> unseen;
  没匹配上: 名字与某个骨干标签有松散重合, 或该角色在 zerochan 上很热门 -> unknown, 否则 unseen.
"""
from __future__ import annotations

import collections
import glob
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPRO = Path('/data/neko/cxip_repro')
MERGED = REPRO / 'data/ccip_merged_nd'
PREMERGE = REPRO / 'data/ccip_merge'
ZC_TAGS = REPRO / 'data/zerochan_m50_800k/tags.parquet'
BACKBONE_TAGS = Path('/data/neko/cxip_v3/models/AnimeDINOv3-timm/tags.json')
ISEK_GLOB = '/data/.hf/hub/datasets--isek-ai--danbooru-tags-2024/snapshots/*/data/*.parquet'
CACHE = Path('/data/neko/cxip_v3/eval/_cache')

V1_CONTEXT = {'fgo': 'fate/grand order', 'arknights': 'arknights', 'azurlane': 'azur lane', 'bluearchive': 'blue archive',
              'genshin': 'genshin impact', 'girlsfrontline': "girls' frontline", 'neuralcloud': 'girls\' frontline: neural cloud'}
# 不是作品名的限定词/目录名
GENERIC_SERIES = {'', 'others', 'female', 'male', 'character', 'vtuber', 'original', 'realperson', 'reallife', 'unknown',
                  'other', 'misc', 'genderbend', 'cosplay', 'mascot', 'oc', 'originalcharacter'}
# FGO 式 "职阶 (真名)" 的职阶不能当名字用
GENERIC_NAMES = {'saber', 'archer', 'lancer', 'rider', 'caster', 'assassin', 'berserker', 'ruler', 'avenger', 'shielder',
                 'foreigner', 'alter ego', 'moon cancer', 'pretender', 'beast', 'servant', 'master', 'unknown',
                 'girl', 'boy', 'female', 'male', 'character', 'protagonist', 'heroine', 'hero', 'player'}
# Danbooru copyright 蕴含关系里不当作父节点的 (会把不相关作品连起来)
NON_PARENT_COPYRIGHTS = {'original', 'real life', 'nintendo', 'sega', 'capcom', 'square enix', 'namco', 'konami',
                         'cygames', 'type-moon', 'key (company)', 'leaf (studio)', 'atlus', 'bandai namco'}


def strip_accents(s: str) -> str:
    s = unicodedata.normalize('NFKD', unicodedata.normalize('NFKC', s))
    return ''.join(ch for ch in s if not unicodedata.combining(ch))


def norm_key(s: str) -> str:
    """作品名的匹配键: 小写、去重音、@->a、去掉前导 the、只留字母数字"""
    s = strip_accents(s).lower().replace('@', 'a').replace('&', 'and')
    s = re.sub(r'[^0-9a-z]+', ' ', s).strip()
    s = re.sub(r'^the ', '', s)
    return s.replace(' ', '')


def norm_name(s: str) -> str:
    """人名的匹配键: 小写、去重音、去撇号, 其他符号变空格"""
    s = strip_accents(s).lower()
    s = re.sub(r"['’`]", '', s)
    s = re.sub(r'[^0-9a-z]+', ' ', s)
    return ' '.join(s.split())


def fuzzy_name(s: str) -> str:
    """罗马字长音折叠: kyouko/kyoko, shuuichi/shuichi, gotou/gotoh/goto, toujou/tojo (Danbooru 与 zerochan 写法不一)"""
    toks = []
    for t in s.split():
        t = re.sub(r'oh$', 'o', t)
        t = t.replace('ou', 'o').replace('oo', 'o').replace('uu', 'u').replace('aa', 'a').replace('ii', 'i').replace('ee', 'e')
        toks.append(t)
    return ' '.join(toks)


def split_quals(s: str):
    """'gilgamesh (caster) (fate)' -> ('gilgamesh', ['caster', 'fate']); 支持嵌套括号 'Foreigner (Abigail Williams (Natsu))'"""
    s = s.strip()
    quals = []
    while s.endswith(')'):
        depth, i = 0, len(s) - 1
        while i >= 0:
            if s[i] == ')':
                depth += 1
            elif s[i] == '(':
                depth -= 1
                if depth == 0:
                    break
            i -= 1
        if i <= 0:
            break
        quals.insert(0, s[i + 1:-1].strip())
        s = s[:i].strip()
    return s, quals


def v2_series(key: str) -> str:
    return re.sub(r'^\d+_\d+_', '', key.split('/')[0]).replace('_', ' ')


# ----------------------------------------------------------------------------------------------------------- Danbooru
class Danbooru:
    """isek-ai/danbooru-tags-2024 的角色 -> copyright 统计 + copyright 蕴含关系 + 骨干角色标签集合"""

    def __init__(self, cache: Path = CACHE, backbone_tags: Path = BACKBONE_TAGS, min_char_posts: int = 5):
        cache.mkdir(parents=True, exist_ok=True)
        if not (cache / 'isek_char_copyright.parquet').exists():
            build_isek_cache(cache)
        cc = pd.read_parquet(cache / 'isek_char_copyright.parquet')
        self.char_count = pd.read_parquet(cache / 'isek_char_count.parquet').set_index('character')['n'].to_dict()
        self.cp_count = pd.read_parquet(cache / 'isek_copyright_count.parquet').set_index('copyright')['n'].to_dict()
        pairs = pd.read_parquet(cache / 'isek_copyright_pairs.parquet')

        tags = json.load(open(backbone_tags))
        self.backbone = {t['name'].replace('_', ' ') for t in tags if t['category_name'] == 'character'}
        for t in self.backbone:  # isek 里改过名的骨干标签 (4 个) 也要能被基名匹配到
            self.char_count.setdefault(t, 0)

        # 角色的 copyright 分布 (只用 <= 2 个角色的帖子统计)
        cc = cc[cc['copyright'] != '']
        tot = cc.groupby('character')['n'].transform('sum')
        cc = cc.assign(share=cc['n'] / tot)
        cc = cc[(cc['share'] >= 0.15) & ~cc['copyright'].isin(['original', 'real life'])].sort_values(['character', 'n'], ascending=[True, False])
        self.char_cp = {c: list(zip(g['copyright'], g['share'])) for c, g in cc.groupby('character')}

        # copyright 蕴含: A -> B 若 P(B|A) >= 0.9 且 B 更大
        pairs['na'] = pairs['a'].map(self.cp_count)
        pairs['nb'] = pairs['b'].map(self.cp_count)
        par = pairs[(pairs['n'] >= 0.9 * pairs['na']) & (pairs['nb'] > pairs['na']) & (pairs['na'] >= 10)
                    & ~pairs['b'].isin(NON_PARENT_COPYRIGHTS)]
        self.parents = collections.defaultdict(list)
        for a, b, nb in zip(par['a'], par['b'], par['nb']):
            self.parents[a].append((nb, b))
        self._root = {}
        # 作品名键 -> Danbooru copyright (同键取帖子最多的)
        self.cp_by_key = {}
        for cp, n in sorted(self.cp_count.items(), key=lambda x: x[1]):
            k = norm_key(cp)
            if k:
                self.cp_by_key[k] = cp
        # 'inazuma eleven (series)' 也能用 'inazumaeleven' 查到 (不覆盖已有的键)
        for cp, n in sorted(self.cp_count.items(), key=lambda x: -x[1]):
            b, q = split_quals(cp)
            if q and q[-1] in ('series', 'game', 'anime', 'manga', 'novel', 'franchise') and n >= 50:
                self.cp_by_key.setdefault(norm_key(b), cp)
        self.big_cp_keys = sorted((k for k, cp in self.cp_by_key.items() if len(k) >= 6 and self.cp_count.get(cp, 0) >= 200),
                                  key=len, reverse=True)
        # 基名 -> 角色标签 (精确 / 词序颠倒 / 长音折叠)
        self.index = {k: collections.defaultdict(list) for k in ('exact', 'swapped', 'fuzzy')}
        for tag, n in self.char_count.items():
            if n < min_char_posts and tag not in self.backbone:
                continue
            base, quals = split_quals(tag)
            b = norm_name(base)
            if not b:
                continue
            self.index['exact'][b].append(tag)
            toks = b.split()
            if 2 <= len(toks) <= 3:
                self.index['swapped'][' '.join(reversed(toks))].append(tag)
            fz = fuzzy_name(b)
            self.index['fuzzy'][fz].append(tag)
            if 2 <= len(toks) <= 3:
                self.index['fuzzy'][' '.join(reversed(fz.split()))].append(tag)
        self.by_base = self.index['exact']

    def root(self, cp: str) -> str:
        """沿蕴含关系走到 franchise 根 (每步取最大的父节点)"""
        if cp in self._root:
            return self._root[cp]
        cur, seen = cp, {cp}
        while self.parents.get(cur):
            nxt = max(self.parents[cur])[1]
            if nxt in seen:
                break
            seen.add(nxt)
            cur = nxt
        self._root[cp] = cur
        return cur

    def tag_franchises(self, tag: str) -> set:
        return {norm_key(self.root(cp)) for cp, _ in self.char_cp.get(tag, [])}

    def char_groups(self, tags, min_posts=20, min_noinfo=50) -> dict:
        """同基名的标签按 franchise 分组 (服装变体 'gawr gura (1st costume)' 与本体同组); 少量帖子的标签忽略"""
        groups = collections.defaultdict(list)
        for t in tags:
            n = self.char_count.get(t, 0)
            if n < min_posts and t not in self.backbone:
                continue
            tf = self.tag_franchises(t)
            if not tf:
                if n >= min_noinfo:
                    groups[('?', t)].append(t)
                continue
            groups[min(tf)].append(t)
        # 一个标签可能有多个 franchise; 有交集的组合并
        keys = list(groups)
        merged = {}
        for k in keys:
            if isinstance(k, tuple):
                merged[k] = groups[k]
                continue
            for t in groups[k]:
                tf = self.tag_franchises(t)
                hit = [m for m in merged if not isinstance(m, tuple) and m & tf]
                if hit:
                    merged[hit[0]] = merged[hit[0]] + [t]
                    merged[hit[0] | tf] = merged.pop(hit[0])
                else:
                    merged[frozenset(tf)] = [t]
        return {k: sorted(set(v)) for k, v in merged.items()}

    def tag_series(self, tag: str) -> list:
        return [cp for cp, _ in self.char_cp.get(tag, [])]


def build_isek_cache(cache: Path = CACHE):
    """isek 全量 (860 万帖) 汇总: 角色计数、copyright 计数、角色-copyright 共现、copyright 两两共现. 约 30 秒"""
    import itertools
    import pyarrow.parquet as pq
    cc, chc, cpc, pc = (collections.Counter() for _ in range(4))
    for f in sorted(glob.glob(ISEK_GLOB)):
        t = pq.read_table(f, columns=['character', 'copyright']).to_pandas()
        for ch, cp in zip(t['character'].values, t['copyright'].values):
            cps = cp.split(', ') if cp else ['']
            for p in cps:
                cpc[p] += 1
            if 1 < len(cps) <= 6:
                for a, b in itertools.permutations(cps, 2):
                    pc[(a, b)] += 1
            if ch is None:
                continue
            chs = ch.split(', ')
            for c in chs:
                chc[c] += 1
                if len(chs) <= 2:
                    for p in cps:
                        cc[(c, p)] += 1
    cache.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([(a, b, v) for (a, b), v in cc.items()], columns=['character', 'copyright', 'n']).to_parquet(
        cache / 'isek_char_copyright.parquet')
    pd.DataFrame(list(chc.items()), columns=['character', 'n']).to_parquet(cache / 'isek_char_count.parquet')
    pd.DataFrame(list(cpc.items()), columns=['copyright', 'n']).to_parquet(cache / 'isek_copyright_count.parquet')
    pd.DataFrame([(a, b, v) for (a, b), v in pc.items()], columns=['a', 'b', 'n']).to_parquet(
        cache / 'isek_copyright_pairs.parquet')


# ------------------------------------------------------------------------------------------------------- class names
@dataclass
class NameCand:
    base: str               # 规范化的人名
    series: set             # 该成员给出的作品键 (norm_key)
    source: str
    raw: str


@dataclass
class ClassInfo:
    label: int
    name: str
    n: dict
    members: list
    cands: list = field(default_factory=list)
    series_raw: list = field(default_factory=list)   # 原始写法 (用于展示)
    series_keys: set = field(default_factory=set)
    zc_count: int = 0


def load_classes():
    return json.load(open(MERGED / 'classes.json'))


def known_series_keys(premerge_classes, db: Danbooru) -> set:
    keys = set(db.cp_by_key)
    for c in premerge_classes:
        if c['source'] == 'v2':
            keys.add(norm_key(v2_series(c['key'])))
    keys |= {norm_key(v) for v in V1_CONTEXT.values()}
    return keys - GENERIC_SERIES


def class_infos(db: Danbooru):
    """每个合并类的名字候选和成员给出的作品"""
    classes = load_classes()
    pre = json.load(open(PREMERGE / 'classes.json'))
    skeys = known_series_keys(pre, db)
    zc_cnt = pd.read_parquet(ZC_TAGS).set_index('name')['count'].to_dict()
    infos = []
    for c in classes:
        info = ClassInfo(label=c['label'], name=c['name'], n=c['n'], members=c['members'])
        for m in c['members']:
            src, key = m['source'], m['key']
            if src == 'zc800k':
                info.zc_count = max(info.zc_count, int(zc_cnt.get(key, 0)))
                base, quals = split_quals(key)
                ser = set()
                name_quals = []
                for q in quals:
                    qk = norm_key(q)
                    if qk in skeys:
                        ser.add(qk)
                        info.series_raw.append(q)
                    elif qk not in GENERIC_SERIES:
                        name_quals.append(q)
                b = norm_name(base)
                if b and b not in GENERIC_NAMES:
                    info.cands.append(NameCand(b, ser, src, key))
                for q in name_quals:  # "Caster (Gilgamesh)" 的真名 / "Moon Cancer (BB Dubai)"
                    qb = norm_name(split_quals(q)[0])
                    if qb and qb not in GENERIC_NAMES:
                        info.cands.append(NameCand(qb, ser, src, key))
            elif src == 'v1':
                ctx, folder = key.split('/', 1)
                ser = set()
                if ctx in V1_CONTEXT:
                    ser.add(norm_key(V1_CONTEXT[ctx]))
                    info.series_raw.append(V1_CONTEXT[ctx])
                b = norm_name(folder)
                if b and b not in GENERIC_NAMES:
                    info.cands.append(NameCand(b, ser, src, key))
            else:  # v2
                sraw = v2_series(key)
                sk = norm_key(sraw)
                ser = set() if sk in GENERIC_SERIES else {sk}
                if ser:
                    info.series_raw.append(sraw)
                leaf = key.rsplit('/', 1)[-1].replace('_', ' ')
                b = norm_name(leaf)
                stoks = norm_name(sraw).split()
                btoks = b.split()
                # "Lancer Fate Grand Order" -> "lancer" (名字末尾带作品名); 也试去掉作品名的最后若干词
                bases = [b]
                for k in range(len(stoks), 0, -1):
                    if len(btoks) > k and btoks[-k:] == stoks[-k:] and k >= min(2, len(stoks)):
                        bases.append(' '.join(btoks[:-k]))
                        break
                for x in list(bases):  # "Berserker Tamamo Cat" -> "tamamo cat"
                    xt = x.split()
                    for j in (2, 1):
                        if len(xt) > j and ' '.join(xt[:j]) in GENERIC_NAMES:
                            rest = ' '.join(xt[j:])
                            if norm_key(rest) != sk:  # "Archer Fate stay night" 去掉职阶后只剩作品名
                                bases.append(rest)
                            break
                for x in dict.fromkeys(bases):
                    if x and x not in GENERIC_NAMES:
                        info.cands.append(NameCand(x, ser, src, key))
        info.series_keys = set()
        for cand in info.cands:
            info.series_keys |= cand.series
        infos.append(info)
    return infos


# ---------------------------------------------------------------------------------------------------------- matching
def series_franchises(keys: set, db: Danbooru, alias: dict) -> set:
    out = set()
    for k in keys:
        k = alias.get(k, k)
        cp = db.cp_by_key.get(k)
        out.add(norm_key(db.root(cp)) if cp else k)
    return out


def resolved_franchises(keys, db: Danbooru, alias: dict) -> set:
    """能对上 Danbooru copyright 的作品键 -> franchise 根"""
    out = set()
    for k in keys:
        k = alias.get(k, k)
        if k in db.cp_by_key:
            out.add(norm_key(db.root(db.cp_by_key[k])))
    return out


def match_class(info: ClassInfo, db: Danbooru, alias: dict):
    """返回 (matched_tags, method, ambiguous_tags).
    - series: 名字 (精确/词序颠倒/长音折叠) 对上且 franchise 相交. 只有能对上 Danbooru copyright 的作品键 (resolved)
      才用来确认/否决; 对不上的 (如 v2 的 VTuber 频道名 Noel_Ch) 视为无作品信息.
    - unique_name: 无可用作品证据时, 多词名字在 Danbooru 中只对应一个角色 (同基名的服装变体按 franchise 归为同一角色)."""
    resolved = resolved_franchises(info.series_keys, db, alias)
    strong, mismatch = set(), set()
    weak = []   # (kind, key, tags)
    for cand in info.cands:
        keys = [('exact', cand.base), ('swapped', cand.base), ('fuzzy', fuzzy_name(cand.base))]
        for kind, key in keys:
            tags = db.index[kind].get(key, [])
            if not tags:
                continue
            ok = [t for t in tags if resolved & db.tag_franchises(t)]
            if ok:
                strong.update(ok)
                continue
            if resolved and any(db.tag_franchises(t) for t in tags):
                mismatch.update(tags)
                # 类的作品与标签冲突: 不作为弱证据 (可能是同名的另一个角色)
                if all(db.tag_franchises(t) for t in tags):
                    continue
            weak.append((kind, key, tags, len(cand.base.split())))
    if strong:
        return sorted(strong), 'series', sorted({t for _, _, ts, _ in weak for t in ts} - strong)[:20]
    unique = {}
    for kind, key, tags, ntok in weak:
        groups = db.char_groups(tags)
        if ntok >= 2 and len(groups) == 1:
            g = next(iter(groups.values()))
            unique[tuple(g)] = g
    weak_tags = {t for _, _, ts, _ in weak for t in ts}
    if len(unique) == 1:
        tags = next(iter(unique.values()))
        return sorted(tags), 'unique_name', sorted(weak_tags - set(tags))[:20]
    if unique or weak_tags:
        return [], 'ambiguous', sorted(weak_tags | mismatch)[:20]
    return [], 'series_mismatch' if mismatch else 'none', sorted(mismatch)[:20]


def learn_series_alias(infos, db: Danbooru, min_votes=3, min_share=0.6) -> dict:
    """非 Danbooru 写法的作品键 -> Danbooru copyright 键: 用该作品下 "名字唯一" 的匹配角色投票"""
    votes = collections.defaultdict(collections.Counter)
    for info in infos:
        for cand in info.cands:
            if len(cand.base.split()) < 2:
                continue
            tags = [t for t in db.by_base.get(cand.base, []) if db.char_count.get(t, 0) >= 20]
            if len(tags) != 1 or not db.char_cp.get(tags[0]):
                continue
            top_cp = db.char_cp[tags[0]][0][0]
            for sk in cand.series:
                if sk not in db.cp_by_key:
                    votes[sk][norm_key(top_cp)] += 1
    alias = {}
    for sk, cnt in votes.items():
        (best, n), tot = cnt.most_common(1)[0], sum(cnt.values())
        if n >= min_votes and n / tot >= min_share:
            alias[sk] = best
    # 前缀: pokemonredgreen -> pokemon, yugioh5ds -> yugioh (取最长的、>= 200 帖的 Danbooru copyright 键)
    all_keys = {k for info in infos for k in info.series_keys}
    for sk in all_keys:
        if sk in db.cp_by_key or sk in alias:
            continue
        for k in db.big_cp_keys:
            if sk.startswith(k) and len(sk) > len(k):
                alias[sk] = k
                break
    return alias


def popularity_threshold(rows, q=0.05):
    """匹配上骨干标签的类里 zerochan 图片数的 q 分位数: 未匹配但比这更热门的 zc 类记 unknown"""
    v = [r['zc_count'] for r in rows if r['backbone_seen'] == 'seen' and r['zc_count'] > 0]
    return float(np.quantile(v, q)) if v else float('inf')


def loose_backbone_hit(info: ClassInfo, db: Danbooru, fr: set) -> list:
    """未匹配的类: 名字与骨干标签有松散重合 (词集合包含 / 单词名同作品) -> 返回这些标签, 用来标 unknown"""
    hits = []
    if not hasattr(db, '_bb_tokens'):
        db._bb_tokens = [(t, set(norm_name(split_quals(t)[0]).split())) for t in db.backbone]
    for cand in info.cands:
        ct = set(cand.base.split())
        if not ct:
            continue
        for t, tt in db._bb_tokens:
            if not tt:
                continue
            if tt == ct or (len(tt) >= 2 and tt <= ct) or (len(ct) >= 2 and ct <= tt):
                hits.append(t)
            elif len(ct & tt) >= 1 and fr and db.tag_franchises(t) & fr and (len(ct) == 1 or len(tt) == 1):
                hits.append(t)
    return sorted(set(hits))


MEMBER_POPULAR = 30   # 没有 zc 成员的类: v1/v2 成员图片数 >= 此值且未匹配 -> unknown (v2-only 且匹配上的类中 30-60 张的 6% 是骨干标签)


def build_class_meta(out_path: Path | None = None):
    """每个合并类一行: 作品、franchise、Danbooru 匹配、backbone_seen (seen / unseen / unknown)"""
    db = Danbooru()
    infos = class_infos(db)
    alias = learn_series_alias(infos, db)
    rows = []
    for info in infos:
        tags, method, amb = match_class(info, db, alias)
        bb = [t for t in tags if t in db.backbone]
        # 匹配标签的 copyright: 类有可用作品信息时只取与之相容的; 否则只取 >= 20 帖且占比 >= 50% 的主 copyright
        resolved = resolved_franchises(info.series_keys, db, alias)
        db_cps = []
        for t in tags:
            cps = db.char_cp.get(t, [])
            if resolved:
                cps = [cp for cp, _ in cps if norm_key(db.root(cp)) in resolved]
            else:
                cps = [cp for cp, sh in cps if sh >= 0.5 and db.char_count.get(t, 0) >= 20]
            db_cps += cps[:1]
        db_series = [cp for cp, _ in collections.Counter(db_cps).most_common()]
        all_keys = set(info.series_keys) | {norm_key(x) for x in db_series}
        franchises = series_franchises(all_keys, db, alias)
        # 主作品: Danbooru copyright > v2 目录 > v1 父目录 > zc 限定词
        raw = collections.Counter(info.series_raw)
        primary = db_series[0] if db_series else (raw.most_common(1)[0][0] if raw else '')
        pf = sorted(series_franchises({norm_key(primary)}, db, alias))[0] if primary else ''
        member_n = max([m['n'] for m in info.members if m['source'] != 'zc800k'] or [0])
        rows.append(dict(
            label=info.label, name=info.name,
            n_train=int(info.n.get('train', 0)), n_val=int(info.n.get('val', 0)), n_test=int(info.n.get('test', 0)),
            n_total=int(sum(info.n.values())), sources=','.join(sorted({m['source'] for m in info.members})),
            n_members=len(info.members), zc_count=info.zc_count, member_n=member_n,
            series=primary, franchise=pf,
            series_raw=sorted(set(info.series_raw)), series_keys=sorted(info.series_keys),
            db_series=db_series, franchises=sorted(franchises),
            danbooru_tags=tags, backbone_tags=bb, match_method=method,
            match_confidence={'series': 'high', 'unique_name': 'medium'}.get(method, ''),
            ambiguous_tags=amb[:20], backbone_seen=('seen' if bb else 'unseen') if tags else None,
            loose_backbone=[], unknown_reason=''))
    thr = popularity_threshold([r for r in rows if r['backbone_seen']])
    for r, info in zip(rows, infos):
        if r['backbone_seen'] is not None:
            continue
        loose = loose_backbone_hit(info, db, set(r['franchises']))
        amb_bb = [t for t in r['ambiguous_tags'] if t in db.backbone]
        r['loose_backbone'] = sorted(set(loose) | set(amb_bb))[:20]
        if r['loose_backbone']:
            r['backbone_seen'], r['unknown_reason'] = 'unknown', 'loose_name_match'
        elif r['zc_count'] >= thr or (r['zc_count'] == 0 and r['member_n'] >= MEMBER_POPULAR):
            r['backbone_seen'], r['unknown_reason'] = 'unknown', 'popular_unmatched'
        else:
            r['backbone_seen'] = 'unseen'
    df = pd.DataFrame(rows)
    aux = dict(series_alias=alias, zc_popularity_threshold=thr, member_popular=MEMBER_POPULAR,
               n_backbone_character_tags=len(db.backbone),
               n_backbone_tags_matched=len({t for x in df['backbone_tags'] for t in x}))
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path)
        json.dump(aux, open(str(out_path).replace('.parquet', '.aux.json'), 'w'), indent=1, ensure_ascii=False)
    return df, aux, db
