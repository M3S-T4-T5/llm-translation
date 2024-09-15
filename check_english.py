import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from utils.llm import LLM_Client
from prompts import TRANSLATION_PROMPT
from utils.language_utils import is_english, translate_text
from tqdm import tqdm
from translator import process_row

data = pd.read_parquet('data/revelio_individual_positions_raw_0.parquet', engine='pyarrow')
cnt = 0
rows = []
for index, row in tqdm(data.iterrows()):
    if not row['title_raw'] or not row['description']:
        continue
    if not is_english(row['title_raw']) or not is_english(row['description']):
        cnt += 1
        rows.append(row)
    if cnt > 10000:
        break

ratio = cnt / len(data)
print(f"Percentage of non-English text: {ratio:.2%}")

# multithreaded translation
print("total rows to translate:", len(rows))
rows = rows[:10000]
llm_client = LLM_Client(service="GROQ")
with ThreadPoolExecutor(max_workers=30) as executor:
    rows = list(tqdm(executor.map(lambda row: process_row(row, llm_client), rows), total=len(rows)))