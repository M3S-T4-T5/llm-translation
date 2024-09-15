import os
import glob
import argparse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import logging

from utils.language_utils import is_english, translate_text
from utils.llm import LLM_Client

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.disable(logging.CRITICAL) # Uncomment to disable logging

def process_row(row, llm_client):
    def translate_if_needed(text, llm_client):
        if text and not is_english(text):
            translated_text = translate_text(text, llm_client)
            logging.info(f"Translated text: {text} -> {translated_text}")
            return translated_text
        return text

    row['title_translated'] = translate_if_needed(row.get('title_raw'), llm_client)
    row['description_translated'] = translate_if_needed(row.get('description'), llm_client)

    return row



def process_parquet_file(file_path, llm_client):
    # Check if the file has been processed
    if file_path.endswith("_processed.parquet"):
        logging.info(f"Skipping already processed file: {file_path}")
        return

    # Load the parquet file
    try:
        df = pd.read_parquet(file_path, engine='pyarrow')
    except Exception as e:
        logging.error(f"Failed to read parquet file {file_path}: {e}")
        return

    logging.info(f"Processing file: {file_path}")
    
    # Process each row with concurrency
    with ThreadPoolExecutor(max_workers=30) as executor:
        rows = list(tqdm(executor.map(lambda row: process_row(row, llm_client), [row for _, row in df.iterrows()]), total=len(df)))

    processed_df = pd.DataFrame(rows)

    # Save the processed file
    output_file = file_path.replace(".parquet", "_processed.parquet")
    try:
        processed_df.to_parquet(output_file)
        logging.info(f"Processed file saved to: {output_file}")
    except Exception as e:
        logging.error(f"Failed to save processed file {output_file}: {e}")


def main(data_directory, specific_files=None):
    """
    Main function to process parquet files in a directory.

    Parameters:
        data_directory (str): Directory containing the parquet files.
        specific_files (list): Specific files to process. If None, process all parquet files in the directory.
    """
    if specific_files:
        files_to_process = [os.path.join(data_directory, file) for file in specific_files]
    else:
        files_to_process = glob.glob(os.path.join(data_directory, "*.parquet"))

    if not files_to_process:
        logging.warning("No parquet files found to process.")
        return

    llm_client = LLM_Client()

    for file_path in files_to_process:
        process_parquet_file(file_path, llm_client)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate columns in Parquet files.")
    parser.add_argument("--data_directory", default="data", type=str, help="Directory containing the Parquet files.")
    parser.add_argument("--files", type=str, nargs="*", help="Specific files to process.")
    
    args = parser.parse_args()
    
    main(args.data_directory, args.files)
