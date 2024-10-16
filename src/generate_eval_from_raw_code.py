from dataclasses import dataclass, field
from colorama import Back, Style
from transformers import HfArgumentParser
from pathlib import Path
import multiprocessing as mp
import os
import re
import numpy as np
import pandas as pd
from typing import Any, List, Dict, Set, Tuple
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)


def merge_strings(docs: List[List[str]], max_length: int) -> List[List[str]]:
    slack_percent = 0.10
    slack_size = int(slack_percent * max_length)

    final = []
    for strings in docs:

        result = []
        current = ""

        for string in strings:
            if current:
                temp_len = len(current) + len(string)
                if temp_len < max_length:
                    # Merge the current string with the accumulated string
                    current += string
                elif len(current) < slack_size or len(string) < slack_size:
                    current += string
                else:
                    result.append(current)
                    current = string
            else:
                # Initialize the accumulation with the first string
                current = string

        # Don't forget to add the last accumulated string to the result
        if current:
            result.append(current)

        final.append(result)
    return final


def chunk_one_df(
    data: pd.DataFrame, chunk_size_characters: int, chunk_overlap: float
) -> pd.DataFrame:
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.SOL,
        chunk_size=chunk_size_characters,
        chunk_overlap=chunk_overlap,
        strip_whitespace=False,
    )

    path_and_chunks = []
    for _, row in data.iterrows():
        path = row["path"]
        content = row["content"]
        texts = [content]
        docs = [splitter.split_text(t) for t in texts]
        chunks = merge_strings(docs, max_length=chunk_size_characters)
        final_chunks = [item for sublist in chunks for item in sublist]
        for fc in final_chunks:
            path_and_chunks.append({"path": path, "chunk": fc})

    return pd.DataFrame(path_and_chunks)


def chunk_and_split_contracts(df_chunk: pd.DataFrame, args: Any) -> pd.DataFrame:
    # Set the seed for numpy RNG
    if args.seed is not None:
        np.random.seed(args.seed)

    MAX_CONTENT_SIZE = int(args.chars_per_token * args.tokens_to_chunk)
    content_chunks = chunk_one_df(df_chunk, MAX_CONTENT_SIZE, MAX_CONTENT_SIZE / 8)

    unique_paths = content_chunks["path"].unique()

    final_result = []

    for path in unique_paths:
        # get content for the path
        content_set: Set[Any] = set(
            content_chunks[content_chunks["path"] == path]["chunk"]
        )

        unique_perms: Set[Tuple[str, str, str, str]] = set()
        while len(unique_perms) < args.samples_per_contract:
            # grab a random chunk to split
            content = np.random.choice(list(content_set))

            # Randomly choose two points in the contract for splitting
            all_lines = content.splitlines()

            if not args.completion_after_newline:
                # split the chunk at random spots
                split_points = tuple(
                    np.sort(
                        np.random.choice(
                            range(1, len(content) - 1), size=2, replace=False
                        )
                    )
                )
                start, end = split_points

                if end - start < args.minimum_split_length:
                    continue

                prefix = content[:start]
                middle = content[start:end]
                suffix = content[end:]

            else:
                # split the chunk at line breaks
                if args.lines_to_mask == 0:
                    # pick random sized line mask
                    split_points = tuple(
                        np.sort(
                            np.random.choice(
                                range(1, len(all_lines) - 1), size=2, replace=False
                            )
                        )
                    )
                else:
                    if args.lines_to_mask > len(all_lines) - 1:
                        continue
                    pt = np.random.choice(range(1, len(all_lines) - args.lines_to_mask))
                    split_points = (
                        pt,
                        pt + args.lines_to_mask,
                    )

                start, end = split_points

                # ensure prior lines are not blank
                if all_lines[start].strip() == "":
                    continue

                prefix = "\n".join(all_lines[:start]) + "\n"
                middle = "\n".join(all_lines[start:end]) + "\n"
                suffix = "\n".join(all_lines[end:])

            # ensure we don't ask the model to predice whitespace only
            if middle.strip() == "":
                continue

            unique_perms.add((path, prefix, middle, suffix))

            # sanity check to exit for small texts
            if not args.completion_after_newline:
                if (
                    len(unique_perms) == len(content) - 1
                ):  # Maximum unique permutations reached
                    break
            else:
                if (
                    len(unique_perms) == len(all_lines) - 1
                ):  # Maximum unique permutations reached
                    break

        for itm in unique_perms:
            final_result.append(
                {
                    "path": str(itm[0]),
                    "prefix": itm[1],
                    "middle": itm[2],
                    "suffix": itm[3],
                }
            )
    return pd.DataFrame(final_result)


# Function to split DataFrame into chunks
def split_dataframe(df: pd.DataFrame, num_chunks: int) -> List[pd.DataFrame]:
    chunk_size = len(df) // num_chunks
    return [df[i : i + chunk_size] for i in range(0, len(df), chunk_size)]


@dataclass
class EvalGenerationArguments:
    raw_code_directory: Path = field(
        metadata={"help": "Directory containing Solidity contracts."}
    )
    output: Path = field(
        metadata={"help": "Output file to save permutations in parquet format."}
    )
    samples_per_contract: int = field(
        default=3, metadata={"help": "Maximum number of samples per contract."}
    )
    completion_after_newline: bool = field(
        default=False,
        metadata={"help": "Make the text to guess start after line breaks"},
    )
    lines_to_mask: int = field(
        default=1, metadata={"help": "Lines to mask from original contract"}
    )
    tokens_to_chunk: int = field(
        default=1024,
        metadata={
            "help": "Approximate faketoken-size chunks to create (1 faketoken=args.chars_per_token)"
        },
    )
    chars_per_token: float = field(
        default=2.0, metadata={"help": "Characters per token to use in calculations"}
    )
    minimum_split_length: int = field(
        default=3,
        metadata={
            "help": "Minimum size of a middle slice that we infer, in characters"
        },
    )

    seed: int = field(
        default=42, metadata={"help": "Seed for RNG to ensure reproducibility."}
    )


def main(args: Any) -> None:
    contracts = [
        item for item in args.raw_code_directory.rglob("*.sol") if item.is_file()
    ]

    print(f"Found {len(contracts)} raw Solidity files")
    print("Validating....")

    data = []
    for item in contracts:
        with open(item, "r") as fl:
            content = fl.read()
            data.append({"path": item, "content": content})

    df = pd.DataFrame(data)

    # Remove files with content less than 3 lines
    df["line_count"] = df["content"].apply(lambda x: len(x.splitlines()))
    df = df[df["line_count"] >= 3]

    # remove content less than 10 chars
    df = df[df["content"].str.len() >= 10]

    whitespace_re = re.compile(r"\s+")

    # Function to normalize content by removing whitespace
    def normalize_content(content):
        return whitespace_re.sub("", content)

    # Remove duplicates by comparing normalized content
    df["normalized_content"] = df["content"].apply(normalize_content)
    df = df.drop_duplicates(subset=["normalized_content"])

    # Drop helper columns
    df = df.drop(columns=["line_count", "normalized_content"])

    print(
        f"Contracts after validation and deduplication: {len(df)} [{len(contracts) - len(df)} removed]"
    )

    num_cores = mp.cpu_count()
    df_chunks = split_dataframe(df, num_cores)
    # Use multiprocessing to process each chunk in parallel
    results = []
    with mp.Pool(num_cores) as pool:
        results = pool.starmap(
            chunk_and_split_contracts, [(chunk, args) for chunk in df_chunks]
        )

    # Merge the results back into a single DataFrame
    df_result = pd.concat(results, ignore_index=True)

    print(f"Created {len(df_result)} chunks")

    # Check if the DataFrame has at least 10 entries
    DEBUG_COUNT = 10
    if len(df_result) >= DEBUG_COUNT:
        # Get random entries
        random_entries = df_result.sample(n=DEBUG_COUNT, random_state=1)
    else:
        # If the DataFrame has fewer than 10 entries, get all entries
        random_entries = df_result.sample(frac=1, random_state=1)

    pd.set_option("display.max_colwidth", None)
    for _, itm in random_entries.iterrows():
        print("---")
        print(f"{itm['prefix']}{Back.GREEN}{itm['middle']}{Style.RESET_ALL}{itm['suffix']}")

    df_result.to_parquet(args.output)


if __name__ == "__main__":
    parser = HfArgumentParser(EvalGenerationArguments)
    args = parser.parse_args_into_dataclasses()[0]

    main(args)
