import pandas as pd

m = pd.read_csv("data/raw/mturk_aggregate.csv")
print("Rows:", len(m))
print("Columns:", m.columns.tolist())
print(m.head(3))

print(type(m.loc[0, "turn_nvc_union"]), m.loc[0, "turn_nvc_union"])