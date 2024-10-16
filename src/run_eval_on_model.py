from dataclasses import dataclass, field
from transformers import HfArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from typing import Any, List, Dict, Set, Tuple, Callable, Optional
import time
from fim_utils import load_template_by_model
import json
from enum import Enum
import os
import sys


class APIProvider(Enum):
    OLLAMA = 1
    OPENAI = 2
    CLAUDE = 3
    HF = 4


def load_model(
    model_name: str,
) -> Tuple[transformers.AutoModel, transformers.AutoTokenizer]:
    dtype = torch.bfloat16

    # load up the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True, device_map="auto"
    )

    # load up the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, trust_remote_code=True
    )

    #if torch.cuda.is_available():
    #    model = model.to("cuda")

    print(f"Loaded local model: {model_name}")

    return model, tokenizer

def get_stop_strings(model):
    model_stop = []
    if "codegemma" in model.lower():
        model_stop = ['<|file_separator|>']
    elif "codestral" in model.lower():
        model_stop = ['[PREFIX]', '[SUFFIX]']
    elif "gpt-" in model.lower():
        model_stop = ['</COMPLETION>']
    elif "claude-" in model.lower():
        model_stop = ['</COMPLETION>']

    return model_stop

def hf_generate_text(
    model: transformers.AutoModel,
    tokenizer: transformers.AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # fix for CodeQwen
    if "token_type_ids" in inputs.keys():
        del inputs["token_type_ids"]

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        do_sample=False,
        top_p=None,
        num_beams=1,
    )

    output_tokens = output.sequences[0]
    input_length = inputs["input_ids"].shape[1]

    # only get the newly generated stuff
    newly_generated_tokens = output_tokens[input_length:]

    if "codegemma" in tokenizer.name_or_path and newly_generated_tokens is not None:

        gemma_file_separator_tok = 70
        gemma_eos_tok = 1

        # trim off combinations of: <eos>, <|file_separator|><eos>, <|file_separator|>

        if (
            len(newly_generated_tokens) > 0
            and newly_generated_tokens[-1] == gemma_eos_tok
        ):
            newly_generated_tokens = newly_generated_tokens[:-1]

        if (
            len(newly_generated_tokens) > 0
            and newly_generated_tokens[-1] == gemma_file_separator_tok
        ):
            newly_generated_tokens = newly_generated_tokens[:-1]

    generated_sequence = tokenizer.decode(
        newly_generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    delta = generated_sequence

    return delta  # Return only the newly generated part


def read_prompt_template(template_path: Optional[Path]) -> Callable[[str, str], str]:
    if template_path is None:
        raise RuntimeError("Asked to read a template but don't know what file")

    txt = template_path.read_text(encoding="utf-8")

    def fill_txt(prefix: str, suffix: str) -> str:
        return txt.format(prefix=prefix, suffix=suffix)

    return fill_txt


def get_supported_template(
    model: str, backup: Path
) -> Optional[Callable[[str, str], str]]:
    try:
        fim_obj = load_template_by_model(model)

        def do_fim_psm(prefix: str, suffix: str) -> str:
            js = fim_obj.psm_txt(pre_txt=prefix, mid_txt=None, suf_txt=suffix)
            d = json.loads(js)
            return d["text"]

        def do_fim_spm(prefix: str, suffix: str) -> str:
            js = fim_obj.spm_txt(pre_txt=prefix, mid_txt=None, suf_txt=suffix)
            d = json.loads(js)
            return d["text"]

        # default to PSM
        which_mode = do_fim_psm
        # Codestral only does SPM
        if "codestral" in model.lower():
            which_mode = do_fim_spm

        return which_mode
    except RuntimeError as re:
        # as a backup, read from text file if specified
        prompt_template = read_prompt_template(backup)
        return prompt_template


def bigmodel_prompt_template(prefix: str, suffix: str) -> str:
    # Template string for the system message and task instructions
    # taken from:
    SYSTEM_MSG = """
You are a HOLE FILLER. You are provided with a file containing holes, formatted as '{{HOLE_NAME}}'. Your TASK is to complete with a string to replace this hole with, inside a <COMPLETION/> XML tag, including context-aware indentation, if needed. All completions MUST be truthful, accurate, well-written and correct.

## EXAMPLE QUERY:

<QUERY>
function sum_evens(lim) {
  var sum = 0;
  for (var i = 0; i < lim; ++i) {
    {{FILL_HERE}}
  }
  return sum;
}
</QUERY>

TASK: Fill the {{FILL_HERE}} hole.

## CORRECT COMPLETION

<COMPLETION>if (i % 2 === 0) {
      sum += i;
    }</COMPLETION>

## EXAMPLE QUERY:

<QUERY>
def sum_list(lst):
  total = 0
  for x in lst:
  {{FILL_HERE}}
  return total

print(sum_list([1, 2, 3]))
</QUERY>

## CORRECT COMPLETION:

<COMPLETION>  total += x</COMPLETION>

## EXAMPLE QUERY:

<QUERY>
// data Tree a = Node (Tree a) (Tree a) | Leaf a

// sum :: Tree Int -> Int
// sum (Node lft rgt) = sum lft + sum rgt
// sum (Leaf val)     = val

// convert to TypeScript:
{{FILL_HERE}}
</QUERY>

## CORRECT COMPLETION:

<COMPLETION>type Tree<T>
  = {$:"Node", lft: Tree<T>, rgt: Tree<T>}
  | {$:"Leaf", val: T};

function sum(tree: Tree<number>): number {
  switch (tree.$) {
    case "Node":
      return sum(tree.lft) + sum(tree.rgt);
    case "Leaf":
      return tree.val;
  }
}</COMPLETION>

## EXAMPLE QUERY:

The 4th {{FILL_HERE}} is Mars.

## CORRECT COMPLETION:

<COMPLETION>planet from the Sun</COMPLETION>

## EXAMPLE QUERY:

function hypothenuse(a, b) {
  return Math.sqrt({{FILL_HERE}}b ** 2);
}

## CORRECT COMPLETION:

<COMPLETION>a ** 2 + </COMPLETION>
"""
    full_prompt = (
        SYSTEM_MSG
        + f"\n\n<QUERY>\n{prefix}{{FILL_HERE}}{suffix}\n</QUERY>\nTASK: Fill the {{FILL_HERE}} hole. Answer only with the CORRECT completion, and NOTHING ELSE. Do it now.\n<COMPLETION>"
    )

    return full_prompt


@dataclass
class EvalArgs:
    model: str = field(
        metadata={
            "help": "Model name or path. Prefix with 'hf/', 'openai/', 'claude/' or end with '.gguf'"
        }
    )
    eval_data_file: Path = field(
        metadata={"help": "A parquet file full of data to run the eval"}
    )
    output: Path = field(
        metadata={
            "help": "A parquet file that also includes predictions from the model"
        }
    )
    prompt_template_path: Optional[Path] = field(
        default=None,
        metadata={"help": "Path to the prompt template file, required for new models."},
    )
    max_new_tokens: int = field(
        default=128, metadata={"help": "Maximum number of new tokens to generate."}
    )


def common_runner(args: Any, generate_model_fn: Any, limit: int = -1):

    df_perms = pd.read_parquet(args.eval_data_file)

    if limit > 0:
        # limit is used for debugging
        df_perms = df_perms[:limit]

    tqdm.pandas(desc="Benchmark Progress")
    result = df_perms.progress_apply(generate_model_fn, axis=1)  # type: ignore

    df_perms["model_guess"] = result.apply(lambda x: x[0])
    df_perms["generation_time"] = result.apply(lambda x: x[1])

    # Save the generated results
    df_perms.to_parquet(args.output)

    print("Sample output:")
    num_samples = min(10, len(df_perms))
    print(df_perms[["middle", "model_guess"]].sample(n=num_samples, random_state=1))


def oai_runner(args: Any, limit: int = -1) -> None:
    from openai import OpenAI

    oai_client = OpenAI()
    model = args.model[7:]
    prompt_template = bigmodel_prompt_template

    def oai_generate_model_guess(row: pd.Series) -> Tuple[str, float]:
        prompt = prompt_template(prefix=row["prefix"], suffix=row["suffix"])

        start_time = time.time()  # Start timing

        response = oai_client.chat.completions.create(
            temperature=0,
            model=model,
            max_tokens=args.max_new_tokens,
            stop=get_stop_strings(model),
            messages=[
                {"role": "user", "content": prompt},
            ],
        )

        generated = response.choices[0].message.content
        if generated.startswith("<COMPLETION>"):
            generated = generated[12:]
        end_time = time.time()  # End timing

        generation_time = end_time - start_time  # Calculate the duration

        return generated, generation_time

    common_runner(args, oai_generate_model_guess, limit)


def claude_runner(args: Any, limit: int = -1) -> None:
    model = args.model[7:]
    from anthropic import Anthropic

    anthropic_client = Anthropic(
        # This is the default and can be omitted
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )
    prompt_template = bigmodel_prompt_template

    def claude_generate_model_guess(row: pd.Series) -> Tuple[str, float]:
        prompt = prompt_template(prefix=row["prefix"], suffix=row["suffix"])

        start_time = time.time()  # Start timing
        message = anthropic_client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=args.max_new_tokens,
            stop_sequences=get_stop_strings(model),
        )

        generated = message.content[0].text
        if generated.startswith("<COMPLETION>"):
            generated = generated[12:]
        end_time = time.time()  # End timing

        generation_time = end_time - start_time  # Calculate the duration

        return generated, generation_time

    common_runner(args, claude_generate_model_guess, limit)


def hf_runner(args: Any, limit: int = -1) -> None:
    model = args.model[3:]
    prompt_template = get_supported_template(model, args.prompt_template_path)
    model, tokenizer = load_model(model)

    def hf_generate_model_guess(row: pd.Series) -> Tuple[str, float]:
        prompt = prompt_template(prefix=row["prefix"], suffix=row["suffix"])

        start_time = time.time()  # Start timing
        generated = hf_generate_text(
            model,
            tokenizer,
            prompt,
            args.max_new_tokens,
        )
        end_time = time.time()  # End timing

        generation_time = end_time - start_time  # Calculate the duration

        return generated, generation_time

    common_runner(args, hf_generate_model_guess, limit)


def ollama_runner(args: Any, limit: int = -1) -> None:
    import ollama

    model = args.model[7:]
    prompt_template = get_supported_template(args.model, args.prompt_template_path)

    def ollama_generate_model_guess(row: pd.Series) -> Tuple[str, float]:
        prompt = prompt_template(prefix=row["prefix"], suffix=row["suffix"])

        start_time = time.time()  # Start timing
        #print(f"Ollama model: {model}")
        #print(f"Ollama prompt: {prompt}")

        model_stop = get_stop_strings(model)

        response = ollama.generate(
            model=model,
            prompt=prompt,
            options={
                "num_predict": args.max_new_tokens,
                "num_ctx": 2048,
                "temperature": 0.0,
                "repeat_penalty": 1.0,
                "penalize_newline": False,
                "stop": model_stop
            },
        )
        end_time = time.time()  # End timing
        #print(f"Ollama response: {response['response']}")
        generated = response["response"]
        generation_time = end_time - start_time  # Calculate the duration

        return generated, generation_time

    common_runner(args, ollama_generate_model_guess, limit)


def main(args: Any) -> None:
    if args.model.startswith("openai/"):
        oai_runner(args)
    elif args.model.startswith("claude/"):
        claude_runner(args)
    elif args.model.startswith("hf/"):
        hf_runner(args)
    elif args.model.startswith("ollama/"):
        ollama_runner(args)
    else:
        raise RuntimeError(f"Unknown model and provider: {args.model}")

    sys.exit(0)


if __name__ == "__main__":
    parser = HfArgumentParser(EvalArgs)
    args = parser.parse_args_into_dataclasses()[0]
    main(args)
