# CCIP 合并数据集 (zc800k + character_similarity v1_pruned + v2)

- 原始: 46899 个类别 (Counter({'v2': 37886, 'zc800k': 5031, 'v1': 3982})), 1876733 张图
- 候选对 15973, codex (gpt-6-astra) 判定: {'same': 10824, 'different': 4949, 'variant': 192, 'unsure': 8}, 未判定 0
- 合并后类别 (训练图 >= 2): 33173; 每类成员数分布: {1: 26653, 2: 4423, 3: 2040, 4: 19, 5: 18, 6: 11, 7: 5, 8: 3, 9: 1}
- 去掉: 与 val/test/评测集重合的训练图 116373, 多人整图 172, 类内重复 609480, 训练图不足的类的图 4617
- 去掉 codex 标出的非单一角色类别 306 个 ({'group_or_species': 43, 'generic_or_collective': 167, 'multiple_characters': 26, 'non_character': 67, 'uncertain': 3}), 4329 张图 (见 dropped_flagged.json; 视觉复核: data/ccip_merge/visual_check.parquet)
- 最终图片: 1141773

| split | v1 | v2 | zc800k | total |
|---|---|---|---|---|
| test | 0 | 0 | 79358 | 79358 |
| train | 91824 | 254406 | 635733 | 981963 |
| val | 0 | 0 | 80452 | 80452 |

- 每类训练图: median 6, mean 29.6, max 821
- variant 对 192: 其中 134 对 (Alter/泳装·换职阶/认得出的年龄或化身变体, 视觉概率 >= 0.2) 已合并, 性转/形态变化等不合并 (见 variants.json)

成员最多的合并类:
- Princess Zelda: zc800k:Princess Zelda, zc800k:Zelda (Breath of the Wild), zc800k:Zelda (Skyward Sword), zc800k:Zelda (Twilight Princess), v1:others/princess_zelda, v2:Zelda_no_Densetsu/Princess_Zelda, v2:Zelda_no_Densetsu_Breath_of_the_Wild/Zelda_Breath_of_the_Wild, v2:Zelda_no_Densetsu_Skyward_Sword/Zelda_Skyward_Sword, v2:Zelda_no_Densetsu_Twilight_Princess/Zelda_Twilight_Princess
- Lancer (Fate/Grand Order): zc800k:Lancer (Fate/Grand Order), zc800k:Caster (Scathach Skadi), zc800k:Ruler (Scathach Skadi), v1:fgo/scathach, v1:fgo/scathach_skadi, v2:Fate_Grand_Order/Lancer_Fate_Grand_Order, v2:Fate_Grand_Order/Lancer_Fate_Grand_Order/Caster_Scathach_Skadi, v2:Fate_Grand_Order/Lancer_Fate_Grand_Order/Caster_Scathach_Skadi/Ruler_Scathach_Skadi
- Archer (Ishtar): zc800k:Archer (Ishtar), zc800k:Space Ishtar, zc800k:Rider (Ishtar), v1:others/archer, v1:others/ishtar, v2:Fate_stay_night/Tohsaka_Rin/Archer_Ishtar, v2:Fate_stay_night/Tohsaka_Rin/Archer_Ishtar/Space_Ishtar, v2:Fate_stay_night/Tohsaka_Rin/Archer_Ishtar/Rider_Ishtar
- Caster (Nitocris): zc800k:Caster (Nitocris), zc800k:Assassin (Nitocris), zc800k:Avenger (Nitocris Alter), v1:fgo/nitocris, v1:others/nitocris, v2:Fate_Grand_Order/Caster_Nitocris, v2:Fate_Grand_Order/Caster_Nitocris/Avenger_Nitocris_Alter, v2:Fate_Grand_Order/Caster_Nitocris/Assassin_Nitocris
- BB (Fate/EXTRA): zc800k:BB (Fate/EXTRA), zc800k:Moon Cancer (BB), zc800k:Moon Cancer (BB Dubai), v1:fgo/bb, v1:others/bb, v2:Fate_EXTRA_CCC/BB_Fate_EXTRA, v2:Fate_EXTRA_CCC/BB_Fate_EXTRA/Moon_Cancer_BB
- Caster (Gilgamesh): zc800k:Caster (Gilgamesh), zc800k:Gilgamesh, v1:fgo/gilgamesh, v1:others/gilgamesh, v2:Fate_stay_night/Gilgamesh, v2:Fate_stay_night/Gilgamesh/Caster_Gilgamesh, v2:Fate_stay_night/Gilgamesh/Gilgamesh_Fate_Prototype
- Caster (Helena Blavatsky): zc800k:Caster (Helena Blavatsky), zc800k:Archer (Helena Blavatsky), v1:fgo/helena_blavatsky, v1:others/helena_blavatsky, v2:Fate_Grand_Order/Caster_Helena_Blavatsky, v2:Fate_Grand_Order/Caster_Helena_Blavatsky/Helena_Blavatsky_Christmas, v2:Fate_Grand_Order/Caster_Helena_Blavatsky/Archer_Helena_Blavatsky
- Ruler (Saint Martha): zc800k:Ruler (Saint Martha), zc800k:Rider (Saint Martha), v1:fgo/martha, v1:others/saint_martha, v2:Fate_Grand_Order/Rider_Saint_Martha, v2:Fate_Grand_Order/Rider_Saint_Martha/Ruler_Saint_Martha, v2:Fate_Grand_Order/Rider_Saint_Martha/Martha_Santa
- Saber (Elizabeth Bathory): zc800k:Saber (Elizabeth Bathory), zc800k:Rider (Elizabeth Bathory), zc800k:Caster (Elizabeth Bathory), v1:others/elizabeth_bathory, v2:Fate_EXTRA_CCC/Lancer_Fate_EXTRA_CCC/Rider_Elizabeth_Bathory, v2:Fate_EXTRA_CCC/Lancer_Fate_EXTRA_CCC/Caster_Elizabeth_Bathory, v2:Fate_EXTRA_CCC/Lancer_Fate_EXTRA_CCC/Saber_Elizabeth_Bathory
- Skadi The Corrupting Heart: zc800k:Skadi The Corrupting Heart, zc800k:Skadi, v1:others/skadi, v1:arknights/skadi, v2:Arknights/Skadi/Skadi_The_Corrupting_Heart, v2:Arknights/Skadi
- Merlin (Fate/stay night): zc800k:Merlin (Fate/stay night), zc800k:Merlin (Fate/Prototype), v1:others/merlin, v1:fgo/merlin, v2:Fate_stay_night/Merlin_Fate_stay_night, v2:Fate_Prototype/Merlin_Fate_Prototype
- Foreigner (Abigail Williams (Natsu)): zc800k:Foreigner (Abigail Williams (Natsu)), zc800k:Foreigner (Abigail Williams), v1:fgo/abigail_williams, v1:others/abigail_williams, v2:Fate_Grand_Order/Foreigner_Abigail_Williams, v2:Fate_Grand_Order/Foreigner_Abigail_Williams/Foreigner_Abigail_Williams_Natsu
- Assassin (Shuten-douji): zc800k:Assassin (Shuten-douji), zc800k:Caster (Shuten-douji), v1:fgo/shuten_douji, v1:others/shuten_douji, v2:Fate_Grand_Order/Assassin_Shuten_douji, v2:Fate_Grand_Order/Assassin_Shuten_douji/Caster_Shuten_douji
- Rider (Marie Antoinette): zc800k:Rider (Marie Antoinette), zc800k:Caster (Marie Antoinette), v1:fgo/marie_antoinette, v1:others/marie_antoinette, v2:Fate_Grand_Order/Rider_Marie_Antoinette, v2:Fate_Grand_Order/Rider_Marie_Antoinette/Caster_Marie_Antoinette
- Assassin (Kama): zc800k:Assassin (Kama), zc800k:Avenger (Kama), v1:others/kama, v2:Fate_stay_night/Matou_Sakura/Assassin_Kama, v2:Fate_stay_night/Matou_Sakura/Assassin_Kama/Avenger_Kama, v2:Fate_stay_night/Matou_Sakura/Assassin_Kama/Kama_Mara
