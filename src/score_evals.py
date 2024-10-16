import pandas as pd
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from codebleu import calc_codebleu
from rapidfuzz import fuzz

@dataclass
class Arguments:
    input_file: str = field(metadata={"help": "Path to the input parquet file"})
    output_file: str = field(metadata={"help": "Path to the parquet file with scores embedded"})
    lines_to_match: int = field(default=0, metadata={"help": "Number of lines to use for scoring. Set 0 for 'same lines as in expected'"})

def compute_indel_distance(reference, candidate, lines):
    metric = fuzz.ratio(reference, candidate)
    return metric / 100.0

def compute_sentence_bleu(reference, candidate, lines):
    # Assuming that both reference and candidate are single sentences
    # and tokenization by words is sufficient
    chencherry = SmoothingFunction()
    sp = reference.splitlines()
    count = len(sp)
    if lines > 0:
        count = min(lines, len(sp))
        reference = "\n".join(sp[:count])

    if count <= 1:
        weights = (1,0,0,0)
    elif count == 2:
        weights = (0.5,0.5,0,0)
    elif count == 3:
        weights = (0.33,0.33,0.33,0)
    else:
        weights = (0.25,0.25,0.25,0.25)

    return sentence_bleu([reference.split()], candidate.split(), weights=weights, smoothing_function=chencherry.method1)

def compute_code_bleu(reference, candidate, lines):
    if lines > 0:
        sp = reference.splitlines()
        reference = "\n".join(sp[: min(lines, len(sp))])

    # ignore dataflow since we do not have enough context to match that on
    # ignore weighed data flow since it doesn't have keywords for Solidity
    # JavaScript is not Solidity but maybe close enough
    result = calc_codebleu([reference], [candidate], lang="javascript", weights=(0.50, 0.00, 0.50, 0.0), tokenizer=None)
    return result['codebleu']

def compute_scored_middle(reference, candidate, lines):
    ref_lines = reference.splitlines()
    can_lines = candidate.splitlines()
    if lines == 0:
        count = min(len(ref_lines), len(can_lines))
    else:
        count = min(len(can_lines), lines)
    new_candidate = "\n".join(can_lines[:count])
    return new_candidate

def process_file(args):
    # Read the parquet file
    df = pd.read_parquet(args.input_file)

    # Compute scores by multiple different metrics

    print("Computing data to score...")
    # step 0: only select the lines to match from the guess
    df['scored_guess'] = df.apply(lambda x: compute_scored_middle(x['middle'], x['model_guess'], args.lines_to_match), axis=1)
    # step 1: only select the lines to match from the actual
    df['scored_middle'] = df.apply(lambda x: compute_scored_middle(x['middle'], x['middle'], args.lines_to_match), axis=1)

    # compute BLEU score
    print("Computing BLEU scores ...")
    df['score_bleu'] = df.apply(lambda x: compute_sentence_bleu(x['scored_middle'], x['scored_guess'], args.lines_to_match), axis=1)
    # compute CodeBLEU score
    print("Computing CODEBLEU scores ...")
    df['score_code_bleu'] = df.apply(lambda x: compute_code_bleu(x['scored_middle'], x['scored_guess'], args.lines_to_match), axis=1)
    # compute Indel score
    print("Computing indel scores ...")
    df['score_indel'] = df.apply(lambda x: compute_indel_distance(x['scored_middle'], x['scored_guess'], args.lines_to_match), axis=1)

    print("Computing score averages ...")
    # Compute average scores per path
    avg_scores = df.groupby('path')[['score_bleu', 'score_code_bleu', 'score_indel']].mean()

    indel_score = avg_scores['score_indel'].sum()
    bleu_score = avg_scores['score_bleu'].sum()
    code_score = avg_scores['score_code_bleu'].sum()

    print(f"Final InDelDist Score   : {indel_score}")
    print(f"Final CodeBleu Score    : {code_score}")
    print(f"Final SentenceBleu Score: {bleu_score}")

    print("Random sampling of scoring data:\n")
    random_rows = df[['scored_middle', 'scored_guess', 'score_bleu', 'score_code_bleu', 'score_indel']].sample(10)
    for index, row in random_rows.iterrows():
        print(f"""---
Middle     : {row['scored_middle']}
Model Guess: {row['scored_guess']}
Score BLEU : {row['score_bleu']}
Score Indel: {row['score_indel']}
Code BLEU  : {row['score_code_bleu']}""")

    df.to_parquet(args.output_file)


if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]
    process_file(args)

