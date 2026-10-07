# CCIP 合并数据集 (zc800k + character_similarity v1_pruned + v2)

- 原始: 46899 个类别 (Counter({'v2': 37886, 'zc800k': 5031, 'v1': 3982})), 1876733 张图
- 候选对 15973, codex (gpt-6-astra) 判定: {'same': 10824, 'different': 4949, 'variant': 192, 'unsure': 8}, 未判定 0
- 合并后类别 (训练图 >= 2): 33243; 每类成员数分布: {1: 26665, 2: 4486, 3: 2090, 4: 2}
- 去掉: 与 val/test/评测集重合的训练图 116373, 多人整图 271, 类内重复 609429, 训练图不足的类的图 4618
- 去掉 codex 标出的非单一角色类别 306 个 ({'group_or_species': 43, 'generic_or_collective': 167, 'multiple_characters': 26, 'non_character': 67, 'uncertain': 3}), 4329 张图 (见 dropped_flagged.json; 视觉复核: data/ccip_merge/visual_check.parquet)
- 最终图片: 1141727

| split | v1 | v2 | zc800k | total |
|---|---|---|---|---|
| test | 0 | 0 | 79358 | 79358 |
| train | 91789 | 254408 | 635722 | 981919 |
| val | 0 | 0 | 80450 | 80450 |

- 每类训练图: median 6, mean 29.5, max 641
- variant 对 (未合并, 见 variants.json): 192

成员最多的合并类:
- Nami: zc800k:Nami, zc800k:Nami (ONE PIECE), v1:others/nami, v2:ONE_PIECE/Nami
- Suzuya: zc800k:Suzuya, zc800k:Suzuya (Kantai Collection), v1:others/suzuya, v2:Kantai_Collection/Suzuya
- Kaito (Vocaloid): zc800k:Kaito (Vocaloid), zc800k:KAITO, v2:VOCALOID/KAITO
- Hatsune Miku: zc800k:Hatsune Miku, v1:others/miku_hatsune, v2:VOCALOID/Hatsune_Miku
- Hu Tao: zc800k:Hu Tao, v1:genshin/hu_tao, v2:Genshin_Impact/Hu_Tao
- Kinomoto Sakura: zc800k:Kinomoto Sakura, v1:others/sakura_kinomoto, v2:Cardcaptor_Sakura/Kinomoto_Sakura
- Raiden Shogun: zc800k:Raiden Shogun, v1:genshin/raiden_shogun, v2:Genshin_Impact/Raiden_Shogun
- Ganyu: zc800k:Ganyu, v1:genshin/ganyu, v2:Genshin_Impact/Ganyu
- Akemi Homura: zc800k:Akemi Homura, v1:others/homura_akemi, v2:Mahou_Shoujo_Madoka_Magica/Akemi_Homura
- Misono Mika: zc800k:Misono Mika, v1:bluearchive/mika, v2:Blue_Archive/Misono_Mika
- Kamisato Ayaka: zc800k:Kamisato Ayaka, v1:genshin/kamisato_ayaka, v2:Genshin_Impact/Kamisato_Ayaka
- Mona Megistus: zc800k:Mona Megistus, v1:genshin/mona, v2:Genshin_Impact/Mona_Megistus
- Sailor Moon (Character): zc800k:Sailor Moon (Character), v1:others/sailor_moon, v2:Bishoujo_Senshi_Sailor_Moon/Tsukino_Usagi/Sailor_Moon_Character
- Rem (Re:Zero): zc800k:Rem (Re:Zero), v1:others/rem, v2:Re_Zero_Kara_Hajimeru_Isekai_Seikatsu/Rem_Re_Zero
- Flandre Scarlet: zc800k:Flandre Scarlet, v1:others/flandre_scarlet, v2:Touhou/Flandre_Scarlet
