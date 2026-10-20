import pandas as pd
import json

label_file="/nfs/danbooru-w640-ws-full-artist/test_split.json"
with open(label_file, 'r') as f:
    label_dict = json.load(f)

df = pd.DataFrame(list(label_dict.items()), columns=["name", "label"])
print(df.shape)

N = 100

cats = pd.Categorical(df[df.columns[-1]])
df['ids'] = cats.codes  # 得到类别id
df_topN = df[df['ids'] < N]

output_file = "/nfs/danbooru-w640-ws-full-artist/test_split_100.json"
subset_dict = df_topN.set_index("name")["label"].to_dict()

with open(output_file, "w") as f:
    json.dump(subset_dict, f, ensure_ascii=False, indent=2)

print(f"Saved {len(subset_dict)} entries to {output_file}")
