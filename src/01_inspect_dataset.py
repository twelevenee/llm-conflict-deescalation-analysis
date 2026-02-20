import pandas as pd

df = pd.read_csv("data/raw/dataset_final.csv")

print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print("\nrelationship_subtype value_counts (top 10):")
print(df["relationship_subtype"].value_counts().head(10))

print("\nrelationship_tag value_counts (top 10):")
print(df["relationship_tag"].value_counts().head(10))

print("\nExample couple rows (relationship_tag):")
df_c = df[df["relationship_subtype"] == "couple"]
print(df_c["relationship_tag"].value_counts().head(10))