import pandas as pd
import numpy as np
import os
from googletrans import Translator
from langdetect import detect
import enchant
import time
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing

dirc = '/data/data/revelio_lab/position_raw/'
file_names = [f for f in os.listdir(dirc) if os.path.isfile(os.path.join(dirc, f))]

start_time1 = time.time()

for i,f in enumerate(file_names):
    num=f.split('_')[-1].split('.')[0]
    df=pd.read_parquet(dirc+f, engine='pyarrow')
    d = enchant.Dict("en_US")
    final_lang_li=[]
    start_time = time.time()
    for phrase in df['title_raw'].astype(str):
        lang=[]
        phrase=phrase.replace(',', ' ')
        phrase=phrase.replace(';', ' ')
        phrase=phrase.replace('|', ' ')
        phrase=phrase.replace('/', ' ')
        phrase=phrase.replace('&', ' ')
        phrase_li=phrase.split(' ')
        phrase_li2=[s for s in phrase_li if s not in ['']]

        for e in phrase_li2:
            if d.check(e):
                lang.append('en')
            else:
                lang.append('ot')
        num_en=lang.count('en')
        try:
            if num_en/len(lang)>0.5:
                final_lang='en'
            else:
                final_lang='ot'
        except:
            final_lang=np.nan
        final_lang_li.append(final_lang)
    df['title_language']=final_lang_li

    
    df_ot=df.loc[df['title_language']=='ot',]

    translator = Translator()
    def my_translate(x):
        time.sleep(0.05)
        try:
            return translator.translate(x, src='auto', dest='en').text
        except:
            return np.nan

    title_input_data=list(df_ot['title_raw'])
    desc_input_data=list(df_ot['description'])
    num_cores = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=20) as pool: 
        title_out = list(tqdm(pool.imap(my_translate, title_input_data), total=len(title_input_data)))
        desc_out = list(tqdm(pool.imap(my_translate, desc_input_data), total=len(desc_input_data)))

    df['title_eng']=df['title_raw']
    df.loc[df['title_language']=='ot','title_eng']=title_out
    df['desc_eng']=df['description']
    df.loc[df['title_language']=='ot','desc_eng']=desc_out

    end_time = time.time()
    total_time = end_time - start_time
    minutes, seconds = divmod(total_time, 60)

    tot_total_time = end_time - start_time1
    minutes_tot, seconds_tot = divmod(tot_total_time, 60)

    # intermediate output
    dir2='/data/data/revelio_lab/position_raw_processed/'
    os.makedirs(dir2, exist_ok=True)

    with open(dir2+"access_output.txt", "w") as f:
        text_data = f"Total time {minutes_tot} minutes and {seconds_tot:.2f} seconds\nProcessed file {i}:\n {minutes} minutes and {seconds:.2f} seconds\n"
        f.write(text_data)

    
    output_path = dir2+'positions_raw_processed_'+num+'.parquet'
    df.to_parquet(output_path,index=False)