"""
为 CSIP v2.1 的 content eraser 挑选要擦除的 tagger 标签: 描述画面内容的标签，以及文字/元数据标签
(签名、水印、用户名、语种、年份等，会成为识别画师的捷径，也要参与 debias)；
去掉描述画风/构图/镜头/光效/背景/画质等属性的标签。

animetimm 的 tagger 只有 general(0) / character(4) / rating(9) 三类，画风、元数据类标签都混在 general 里，
所以 general 标签按 Danbooru 的 tag_group wiki (deepghs/danbooru_wikis_full 的 wiki_pages.csv) 分节归类，
再加上命名规则和人工补充的列表。character / rating 视为内容。
优先级: character/rating > 文字/元数据 > 排除规则 > 默认算内容

输出 csv 每行对应 tagger 的一个输出 (顺序与 selected_tags.csv 一致)，列: index, tag_id, name, category, content, reason
FrozenTagger(repo_id, tag_file=...) 读取 content==1 的 index 作为 content eraser 的监督目标。

用法:
  python scripts/tagger/content_tags.py --repo animetimm/caformer_b36.dbv4-full \
    --output data/content_tags/caformer_b36.dbv4-full.csv
"""
import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

CAT_GENERAL, CAT_CHARACTER, CAT_RATING = 0, 4, 9

# tag_group -> 排除的小节 (None 表示整组排除)
EXCLUDE_SECTIONS = {
    'image_composition': ['View Angle', 'Perspective / Depth', 'Composition', 'Format', 'Framing the body',
                          'Styles', 'Techniques', 'Flaws'],  # 保留 Subject matter (各种 focus)、花纹
    'backgrounds': None,
    'colors': None,
    'metatags': None,
    'scan': None,
    'drawing_software': None,
    'lighting': ['Directional', 'Types', 'See also'],  # 保留 Light sources (灯、火把等实物) 和 Misc
}
# tag_group -> 只保留的小节，其余排除
KEEP_SECTIONS = {
    'censorship': ['Diegetic censorship'],  # 头发/蒸汽/手臂遮挡是画面里的东西; 马赛克、黑条等后期打码不是
}

# 文字/元数据: 参与 debias。优先于排除规则 (watermark、web_address、logo 等同时在 image_composition 的 Flaws/Format 里)
TEXT_META_SECTIONS = {
    'text': ['Image metadata', 'Literary devices', 'Specific pieces of text', 'Asian languages',
             'Germanic / Celtic languages', 'Romance languages', 'Balto-Slavic languages',
             'Other Indo-European languages', 'Uralic languages', 'Turkic languages',
             'Other writing systems and languages, including fictional ones:', 'Errors', 'Translation', 'Other'],
    # Misc 里混着 sound_effects、dialogue_box (漫画元素)、text_background (背景)、silent_comic (格式)，下面单独挑
    'year_tags': None,
}
TEXT_META_RULES = [
    (r'^(19|20)\d\d$', 'year'),
    (r'_username$|watermarks?$', 'metadata'),
]
TEXT_META_EXPLICIT = {
    'text': ['wall_of_text', 'text_focus', 'text-only_page', 'foreground_text', 'gradient_text', 'rainbow_text',
             'pixel_text', 'backwards_text', 'mirrored_text', 'page_number', 'textless_version', 'detexted'],
    'metadata': [  # 平台/画师 logo; chaldea_logo 这类画面里的徽标本来就算内容
        'artist_logo', 'patreon_logo', 'twitter_logo', 'twitter_x_logo', 'instagram_logo', 'pixiv_logo',
        'facebook_logo', 'deviantart_logo', 'fanbox_logo', 'weibo_logo', 'youtube_logo', 'tiktok_logo',
    ],
}

# (正则, 原因)，只作用于 general
NAME_RULES = [
    (r'_\(style\)$|artstyle', 'style'),
    (r'_\(texture\)$', 'style'),
    (r'_outline$', 'style'),  # 贴纸式描边
    (r'_\(medium\)$', 'medium'),
    (r'_\(software\)$', 'software'),
    (r'_background$', 'background'),
    (r'_border$|(^|_)inset$', 'layout'),
    (r'speech_bubble$|^thought_bubble$|^spoken_', 'comic'),
    (r'^pov_|_out_of_frame$', 'framing'),
]

# 人工补充 (wiki 分组没覆盖到的)，只作用于 general
EXPLICIT = {
    'style': [
        'retro_artstyle', 'official_style', 'flat_color', 'anime_coloring', 'cel_shading', 'screentones',
        'color_trace', 'paper_texture', 'faux_figurine', 'fine_fabric_emphasis', 'semi-realistic',
        'chibi', 'chibi_only',
        # 五官画法: 有没有画鼻子/嘴唇/睫毛主要取决于画师
        'dot_nose', 'no_nose', 'dot_mouth', 'solid_oval_eyes', 'solid_circle_eyes', 'blush_stickers',
        'lips', 'nose', 'eyelashes', 'long_eyelashes', 'thick_eyelashes', 'shiny_skin', 'shiny_hair',
    ],
    'effect': [
        'light_rays', 'light_particles', 'sunbeam', 'dappled_sunlight', 'sparkle', 'glint', 'light_trail',
        'window_shadow', 'colored_shadow', 'screen_light', 'rim_lighting', 'underlighting',
    ],
    'comic': [
        'notice_lines', 'sound_effects', 'dialogue_box', 'splash_page', 'square_4koma', 'segmented_comic',
    ],
    'layout': [
        'split_screen', 'split_theme', 'variations', 'expression_chart', 'height_chart', 'bust_chart',
        'character_chart', 'art_program_in_frame', 'art_tools_in_frame',
    ],
    'framing': ['three_quarter_view', 'blurry_foreground', 'soft_focus'],
    'flaw': ['jaggy_lines', 'bad_perspective', 'censored_identity', 'pixelated'],
}


def norm_tag(s):
    return s.strip().lower().replace(' ', '_')


def parse_tag_group(body):
    '''DText: "h4#id. Section" 是小节标题, [[tag]] / [[tag|text]] 是标签'''
    sections, cur = {}, '_head'
    for line in body.replace('\r', '').split('\n'):
        m = re.match(r'^h\d[^.]*\.\s*(.*)$', line)
        if m:
            cur = m.group(1).strip()
            continue
        for t in re.findall(r'\[\[([^\]|]+)', line):
            if not t.lower().startswith('tag group'):
                sections.setdefault(cur, []).append(norm_tag(t))
    return sections


def build_group_tags(wiki_csv):
    '''返回 (文字/元数据标签, 排除的标签)，值为来源 group/section'''
    wiki = pd.read_csv(wiki_csv, usecols=['title', 'body'])
    wiki = wiki[wiki['title'].str.startswith('tag_group:', na=False)].set_index('title')['body']

    def sections(group):
        title = f'tag_group:{group}'
        if title not in wiki.index:
            raise KeyError(f'{title} not found in {wiki_csv}')
        return parse_tag_group(wiki[title]).items()

    text_meta, excl = {}, {}
    for group, secs in TEXT_META_SECTIONS.items():
        for sec, tags in sections(group):
            if secs is None or sec in secs:
                for t in tags:
                    text_meta.setdefault(t, f'group:{group}/{sec}')
    for group in [*EXCLUDE_SECTIONS, *KEEP_SECTIONS]:
        for sec, tags in sections(group):
            if group in KEEP_SECTIONS:
                drop = sec not in KEEP_SECTIONS[group]
            else:
                drop = EXCLUDE_SECTIONS[group] is None or sec in EXCLUDE_SECTIONS[group]
            if drop:
                for t in tags:
                    excl.setdefault(t, f'group:{group}/{sec}')
    return text_meta, excl


def match_text_meta(name, group_text_meta):
    if name in group_text_meta:
        return group_text_meta[name]
    for r, ts in TEXT_META_EXPLICIT.items():
        if name in ts:
            return f'manual:{r}'
    for pat, r in TEXT_META_RULES:
        if re.search(pat, name):
            return f'rule:{r}'
    return None


def classify(df, group_text_meta, group_excl, keep_character=True, keep_rating=True, keep_text_meta=True):
    explicit = {t: r for r, ts in EXPLICIT.items() for t in ts}
    rows = []
    for idx, (name, cat) in enumerate(zip(df['name'], df['category'])):
        tm = match_text_meta(name, group_text_meta) if cat == CAT_GENERAL else None
        if cat == CAT_CHARACTER:
            content, reason = keep_character, 'character'
        elif cat == CAT_RATING:
            content, reason = keep_rating, 'rating'
        elif tm is not None:
            content, reason = keep_text_meta, f'text_meta|{tm}'
        elif name in group_excl:
            content, reason = False, group_excl[name]
        elif name in explicit:
            content, reason = False, f'manual:{explicit[name]}'
        else:
            content, reason = True, 'general'
            for pat, r in NAME_RULES:
                if re.search(pat, name):
                    content, reason = False, f'rule:{r}'
                    break
        rows.append((idx, name, cat, int(content), reason))
    out = pd.DataFrame(rows, columns=['index', 'name', 'category', 'content', 'reason'])
    if 'tag_id' in df.columns:
        out.insert(1, 'tag_id', df['tag_id'].values)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', default='animetimm/caformer_b36.dbv4-full', help='animetimm tagger repo')
    parser.add_argument('--wiki-csv', type=Path, default=None,
                        help='danbooru wiki_pages.csv, 默认从 deepghs/danbooru_wikis_full 下载')
    parser.add_argument('--output', type=Path, default=Path('data/content_tags/caformer_b36.dbv4-full.csv'))
    parser.add_argument('--drop-character', action='store_true', help='角色标签也不作为内容 (v2.1 另有 char eraser)')
    parser.add_argument('--drop-rating', action='store_true')
    parser.add_argument('--drop-text-meta', action='store_true', help='文字/元数据标签不参与 debias')
    args = parser.parse_args()

    df = pd.read_csv(hf_hub_download(repo_id=args.repo, repo_type='model', filename='selected_tags.csv'),
                     keep_default_na=False)
    wiki_csv = args.wiki_csv or hf_hub_download(repo_id='deepghs/danbooru_wikis_full', repo_type='dataset',
                                                filename='wiki_pages.csv')
    out = classify(df, *build_group_tags(wiki_csv), keep_character=not args.drop_character,
                   keep_rating=not args.drop_rating, keep_text_meta=not args.drop_text_meta)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f'{args.repo}: {len(out)} tags, content: {out["content"].sum()} -> {args.output}')
    print(out.groupby('category')['content'].agg(['size', 'sum']).rename(columns={'size': 'total', 'sum': 'content'}))
    for flag in [1, 0]:
        print('content' if flag else 'excluded')
        reasons = Counter(r.split('/')[0] for r in out.loc[out['content'] == flag, 'reason'])
        for r, n in reasons.most_common():
            print(f'  {r}: {n}')


if __name__ == '__main__':
    main()
