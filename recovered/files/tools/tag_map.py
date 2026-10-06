import pandas as pd

csv_full = '/nfs/danbooru-wdtagger-v4-w640-ws-30k/selected_tags.csv'
csv_small = '/nfs/danbooru-wdtagger-v4-w640-ws-30k/tags.csv'

df_full = pd.read_csv(csv_full)
df_small = pd.read_csv(csv_small)

# 创建从tag_id到df_full行索引的映射
tag_id_to_index = pd.Series(df_full.index, index=df_full['tag_id'])

# 为df_small的每一行找到在df_full中对应的index
indices = df_small['tag_id'].map(tag_id_to_index)

# 创建包含tag_id、name和对应index的DataFrame
result_df = pd.DataFrame({
    'tag_id': df_small['tag_id'],
    'name': df_small['name'],
    'index_in_full': indices.astype('Int64')  # 使用可空整数类型，NaN保持为NaN
})

# 保存到CSV文件
output_path = '/nfs/danbooru-wdtagger-v4-w640-ws-30k/tags_map.csv'
result_df.to_csv(output_path, index=False)

print(f"结果已保存到: {output_path}")
print(f"共 {len(result_df)} 行数据")
