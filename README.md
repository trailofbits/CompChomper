# CompChomper: Code Completion Evalution

CompChomper is a framework for measuring how LLMs perform at code completion.

CompChomper works by ingesting Github repositories that *you* want to use as an evaluation set and using them to measure completion abilities of popular LLMs.

The following models are supported by CompChomper:
* Claude (3.0 Sonnet, 3.0 Opus, 3.5 Sonnet)
* CodeGemma
* CodeLLama
* CodeQwen
* Codestral
* GPT {3.5, 4, 4o}
* DeepSeek Coder (v1.5, v2)
* StarCoder2

The following inference engines are supported:
* Claude API (only for Claude models)
* OpenAI API (only for OpenAI models)
* Ollama
* HuggingFace Transformers

We used CompChomper to measure how well different LLMs perform at completing Solidity code. It can be easily modified to handle other programming languages.

**This work should be considered research/proof-of-concept quality.** We have tried to document our choices and script most actions to enable repeatability.

# Solity Evaluation Details

No evaluation is perfect, and neither is this one. It makes some arbitrary choices and likely has bugs.

We have [written down some of our thought regarding evalution choices for measuring completion of Solidity code](docs/SolidityEvaluation.md).

## Some evaluation notes

The large commercial models don’t expose a FIM interface over API, so we used prompts borrowed from Continue.dev to simulate code infill capability.

How do we grade similarity? This is a tough question! Imagine the expected value of a completion is `foo(bar)`. There are two models, A and B. Model A answers `foo( bar )` and model B answer `frob(bar)`. These answers are equally distant (two edits apart), but model A is clearly better. We evaluated three measurements: BLEU, CodeBLEU and Insert/Delete (Indel) Distance as possible metrics. We present the Indel scores as these felt the most useful and consistent, even though in this case it would not differentiate the models.

# Quickstart Guide

This quickstart guide shows you how to reproduce our results evaluating Solidity code completion abilities of various LLMs. There is guidance for running the evaluations remotely on Modal Labs (recommended) or locally if you have an appropriate GPU or are working with another cloud provider. 

## Local Instructions

```sh
# Install all pre-requisites & setup a virtual environment
make dev

#set up OPENAI_API_KEY, HF_TOKEN, ANTHROPIC_API_KEY in your env file:
vim .env
# <add the stuff>

# Fetch evaluation code and convert it into packages
make eval_packages_2024-06

# fetch models and ensure API keys are setup
make fetch_all

# Evaluate on single file mode
make eval_all EVAL_DATA_FILE=eval_packages/eval_2024-06_whole_line.parquet
make eval_all EVAL_DATA_FILE=eval_packages/eval_2024-06_until_eol.parquet

# score every output (using only 1 line to match equivalence)
for i in eval_outputs/*.parquet; do
    python3 src/score_evals.py --input_file $i --output_file eval_scores/$(basename $i) --lines_to_match 1 2>/dev/null
done

# Display the output in a nice table
python3 src/output_table.py eval_scores/*.parquet
```


## Remote Evaluation (Via Modal Labs)

```sh
# Install all pre-requisites & setup a virtual environment
make dev

# Install modal labs cli tool
python3 -m pip install modal
python3 -m modal setup

#set up OPENAI_API_KEY, HF_TOKEN, ANTHROPIC_API_KEY in your env file:
vim .env
# <add the stuff>

#Create the Eval Packages
modal run remote_eval.py::make_eval_packages

# Sanity check: fetch all the models and validate API key environment
modal run remote_eval.py::fetch_all

export EVAL_DATA_FILE="eval_packages/eval_2024-06_whole_line.parquet;eval_packages/eval_2024-06_until_eol.parquet"

# evaluate HF models
modal run remote_eval.py::eval_hf

# evaluate ollama models
modal run remote_eval.py::eval_ollama

# evaluate openai models
modal run remote_eval.py::eval_openai

# evaluate claude models
modal run remote_eval.py::eval_claude

# copy outputs from `eval_outputs` modal volume to local `eval_outputs/` directory
modal volume get --force eval_outputs / eval_outputs/

# score every output (using only 1 line to match equivalence)
for i in eval_outputs/*.parquet; do
    python3 src/score_evals.py --input_file $i --output_file eval_scores/$(basename $i) --lines_to_match 1 2>/dev/null
done

# emit a nice table
python3 src/output_table.py eval_scores/*.parquet
