import os
import requests
import pandas as pd
from io import StringIO

# Load raw data
URL = 'https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-11-03/ikea.csv'

def get_data_ikea(url: str, local_path: str = 'IZStepProjectPython/ikea.csv', timeout: int = 30) -> pd.DataFrame | None:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
    except Exception as e:
        print(f'[ERROR] Load/parse failed: {e}')
        return None

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        df.to_csv(local_path, index=False, encoding='utf-8')
    except OSError as e:
        print(f'[WARN] Couldn\'t save raw CSV: {e}')

    return df

# Cleaning pipeline

df = get_data_ikea(URL)
if df is None:
    raise SystemExit('Failed to load data.')

# 1) Best row per item_id (max non-null fields)
# df['_non_null_cnt'] = df.count(axis=1)
df['_non_null_cnt'] = df.notna().sum(axis=1)
df = (
    df.sort_values(['item_id', '_non_null_cnt'], ascending=[True, False])
      .drop_duplicates('item_id', keep='first')
      .drop(columns=['_non_null_cnt'])
      .reset_index(drop=True)
)

# 2) Coerce numeric types for core fields (old_price is kept but ignored)
for col in ['depth', 'height', 'width', 'price']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 3) Clean designer + normalized variant
mask_err = df['designer'].str.match(r'^\d{3}\.\d{3}\.\d{2}', na=False) | (df['designer'].str.len() > 150)
df['description_from_designer'] = pd.NA
df.loc[mask_err, 'description_from_designer'] = df.loc[mask_err, 'designer']
df.loc[mask_err, 'designer'] = pd.NA

def normalize_designer(name: str) -> str:
    if pd.isna(name) or name == 'Unknown':
        return name
    parts = sorted(p.strip() for p in str(name).split('/'))
    return ' / '.join(parts)

df['designer_norm'] = df['designer'].map(normalize_designer)

# 4) Split by presence of all dimensions
dims = ['depth', 'height', 'width']
mask_full = df[dims].notna().all(axis=1)
df_full = df.loc[mask_full].copy()
df_miss = df.loc[~mask_full].copy()

print(mask_full.value_counts())
print(df_full.shape, df_miss.shape)

# 5) Drop dups by designer_norm + dimensions
subset_dims = ['designer_norm'] + dims
dups_dims_n = df_full.duplicated(subset=subset_dims, keep=False).sum()
df_full = df_full.sort_values(subset_dims).drop_duplicates(subset=subset_dims, keep='first')
print(f'Size-dup rows found: {dups_dims_n}')

# 6) Merge back
df = pd.concat([df_full, df_miss], ignore_index=True)

# 7) Strict content dups (ignore old_price)
subset_strict = ['name', 'price', 'designer_norm', 'depth', 'height', 'width', 'short_description']
mask_full2 = df[dims].notna().all(axis=1)
df_full2 = df.loc[mask_full2].copy()
dups_strict_n = df_full2.duplicated(subset=subset_strict, keep=False).sum()
df_full2 = df_full2.drop_duplicates(subset=subset_strict, keep='first')
df = pd.concat([df_full2, df.loc[~mask_full2]], ignore_index=True)
print(f'Strict-dup rows found: {dups_strict_n}')

# 8) Remove simple outliers: non-positive sizes
before = len(df)
mask_pos = (
    (df['depth'].gt(0) | df['depth'].isna()) &
    (df['height'].gt(0) | df['height'].isna()) &
    (df['width'].gt(0) | df['width'].isna())
)
df = df.loc[mask_pos].reset_index(drop=True)
print(f'Removed non-positive sizes: {before - len(df)} rows')

# Save cleaned dataset
out_path = 'IZStepProjectPython/ikea_clean.csv'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df.to_csv(out_path, index=False, encoding='utf-8')
print(f'[Done] Saved: {out_path}  |  shape={df.shape}')


# Example inspection
for c in dims:
    print(c, "== 1:", (df[c] == 1).sum(), "| non-null:", df[c].notna().sum())

mask_ones = (df[dims] == 1).any(axis=1)
df.loc[mask_ones, ['name', 'category', 'price'] + dims].head(30)

for c in dims:
    print("\n", c)
    print(df[c].dropna().value_counts().head(10))  
    print("min:", df[c].min(), "5 smallest:", df[c].dropna().nsmallest(5).tolist())


for c in dims:
    print(c, "<=1:", (df[c] <= 1).sum(), "| <=2:", (df[c] <= 2).sum())