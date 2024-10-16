import argparse
import pandas as pd
import os
import re

def process_file(file_path, eval_name):
    df = pd.read_parquet(file_path)
    avg_scores = df.groupby('path')[['score_bleu', 'score_code_bleu', 'score_indel']].mean()

    indel_score = avg_scores['score_indel'].sum()
    bleu_score = avg_scores['score_bleu'].sum()
    code_score = avg_scores['score_code_bleu'].sum()

    return {
        "Name": os.path.basename(file_path).replace("output_", "").replace("_eval_", "_").replace(".parquet", "").replace(f"_{eval_name}", ""),
        "BLEU": bleu_score,
        "CodeBLEU": code_score,
        "Indel": indel_score
    }

def main(file_list):
    eval_tables = {}

    for file in file_list:
        match = re.match(r'output_(.+)_eval_(.+)\.parquet', os.path.basename(file))
        if match:
            provider_model = match.group(1)
            eval_name = match.group(2)

            if eval_name not in eval_tables:
                eval_tables[eval_name] = []

            result = process_file(file, eval_name)
            eval_tables[eval_name].append(result)

    for eval_name, results in eval_tables.items():
        df = pd.DataFrame(results)
        print(f"Table: eval_{eval_name}")
        print(df.to_string(index=False))
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process parquet files to output evaluation tables.")
    parser.add_argument('files', metavar='F', type=str, nargs='+', help='List of parquet files')
    args = parser.parse_args()

    main(args.files)

