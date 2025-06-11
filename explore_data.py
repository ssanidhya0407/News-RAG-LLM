import json
import pandas as pd

# Path to your dataset
DATA_PATH = "data/News_Category_Dataset_v3.json"

# Load the JSON lines dataset into a list
records = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

# Convert to DataFrame
df = pd.DataFrame(records)

print("Number of records:", len(df))
print("Columns:", df.columns.tolist())
print("Sample rows:")
print(df.head(3).to_string(index=False))