import pandas as pd
from utils.language_utils import is_english
from tqdm import tqdm

data = pd.read_parquet('data/revelio_individual_positions_raw_0.parquet', engine='pyarrow')
cnt = 0
for index, row in tqdm(data.iterrows()):
    if not row['title_raw'] or not row['description']:
        continue
    if not is_english(row['title_raw']) or not is_english(row['description']):
        cnt += 1
    
ratio = cnt / len(data)
print(f"Percentage of non-English text: {ratio:.2%}")