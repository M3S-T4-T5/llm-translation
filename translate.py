import os
import pandas as pd
from tqdm import tqdm
from utils import is_english

pd.options.display.max_columns = 999

DATA_DIR = "/storage/yunhan/revelio_lab/position_raw_processed/"

def main():
    data = pd.read_parquet(DATA_DIR + "positions_raw_processed_143000000.parquet")


    cnt = 0
    total_cnt = len(data)
    for index, row in tqdm(data.iterrows()):
        # check null
        if pd.isnull(row['title_raw']) or pd.isnull(row['description']):
            continue
        if is_english(row['title_raw']) and is_english(row['description']):
            continue
        cnt += 1
    
        
    print("Non-English text ratio: ", cnt / total_cnt)
    print(data.head())
    # print(non_english_data[:5])
    



if __name__ == '__main__':
    main()