#### Begin Dev Env Setup
# Optionally overriden by the user, if they're using a virtual environment manager.
VENV ?= env

# On Windows, venv scripts/shims are under `Scripts` instead of `bin`.
VENV_BIN := $(VENV)/bin
ifeq ($(OS),Windows_NT)
		VENV_BIN := $(VENV)/Scripts
endif

.PHONY: all
all:
		@echo "Run my targets individually!"

.PHONY: dev
dev: $(VENV)/pyvenv.cfg
		@. $(VENV_BIN)/activate

$(VENV)/pyvenv.cfg: requirements.txt
		# Create our Python 3 virtual environment
		python3 -m venv env
		$(VENV_BIN)/python -m pip install --upgrade pip setuptools
		$(VENV_BIN)/python -m pip install -r requirements.txt

#### End Dev Env Setup

# File location
CONFIG_FILE := config/models.json

.PHONY: fetch_all fetch_hf fetch_ollama fetch_claude fetch_openai eval_all eval_hf eval_ollama eval_claude eval_openai
.PHONY: eval_packages_2024-06


eval_raw_files/eval_2024-06: config/eval_set_2024-06.json
	python3 src/fetch_eval_files.py $<

eval_packages_2024-06: eval_packages/eval_2024-06_whole_line.parquet eval_packages/eval_2024-06_until_eol.parquet

eval_packages/eval_2024-06_whole_line.parquet: eval_raw_files/eval_2024-06
	python3 src/generate_eval_from_raw_code.py \
		--raw_code_directory $< \
		--output $@ \
		--samples_per_contract 3 \
		--completion_after_newline True \
		--lines_to_mask 1 \
		--seed 42

eval_packages/eval_2024-06_until_eol.parquet: eval_raw_files/eval_2024-06
	python3 src/generate_eval_from_raw_code.py \
		--raw_code_directory $< \
		--output $@ \
		--samples_per_contract 3 \
		--completion_after_newline False \
		--minimum_split_length 3 \
		--lines_to_mask 1 \
		--seed 42

# Main command to fetch all models
fetch_all: fetch_hf fetch_ollama fetch_claude fetch_openai

# Fetch models from Hugging Face
fetch_hf:
ifndef HF_TOKEN
	$(error HF_TOKEN is not set)
endif
	@echo "Fetching Hugging Face models..."
	@jq -r '.[] | select(.provider=="hf") | .models[]' $(CONFIG_FILE) | xargs -I {} huggingface-cli download --repo-type model {}

# Fetch models from Ollama
fetch_ollama:
	@echo "Fetching Ollama models..."
	@jq -r '.[] | select(.provider=="ollama") | .models[]' $(CONFIG_FILE) | xargs -I {} ollama pull {}

# Fetch models from Claude
fetch_claude:
ifndef ANTHROPIC_API_KEY
	$(error ANTHROPIC_API_KEY is not set)
endif
	@echo "Environment variable ANTHROPIC_API_KEY is set, ready to use Claude models."

# Fetch models from OpenAI
fetch_openai:
ifndef OPENAI_API_KEY
	$(error OPENAI_API_KEY is not set)
endif
	@echo "Environment variable OPENAI_API_KEY is set, ready to use OpenAI models."

# Main command to evaluate all models
eval_all: eval_hf eval_ollama eval_claude eval_openai

# Evaluate models from Hugging Face
eval_hf:
ifndef EVAL_DATA_FILE
	$(error EVAL_DATA_FILE is not set)
endif
	@echo "Evaluating Hugging Face models..."
	@jq -r '.[] | select(.provider=="hf") | .models[]' $(CONFIG_FILE) | \
	xargs -I {} sh -c 'python3 src/run_eval_on_model.py --model hf/{} --eval_data_file $$EVAL_DATA_FILE --output "eval_outputs/output_hf_$$(basename {})_$$(basename $$EVAL_DATA_FILE)"'


# Evaluate models from Ollama
eval_ollama:
ifndef EVAL_DATA_FILE
	$(error EVAL_DATA_FILE is not set)
endif
	@echo "Evaluating Ollama models..."
	@jq -r '.[] | select(.provider=="ollama") | .models[]' $(CONFIG_FILE) | \
	xargs -I {} sh -c 'python3 src/run_eval_on_model.py --model ollama/{} --eval_data_file $$EVAL_DATA_FILE --output "eval_outputs/output_ollama_$$(basename {})_$$(basename $$EVAL_DATA_FILE)"'

# Evaluate models from Claude
eval_claude:
ifndef EVAL_DATA_FILE
	$(error EVAL_DATA_FILE is not set)
endif
	@echo "Evaluating Claude models..."
	@jq -r '.[] | select(.provider=="claude") | .models[]' $(CONFIG_FILE) | \
	xargs -I {} sh -c 'python3 src/run_eval_on_model.py --model claude/{} --eval_data_file $$EVAL_DATA_FILE --output "eval_outputs/output_claude_$$(basename {})_$$(basename $$EVAL_DATA_FILE)"'

# Evaluate models from OpenAI
eval_openai:
ifndef EVAL_DATA_FILE
	$(error EVAL_DATA_FILE is not set)
endif
	@echo "Evaluating OpenAI models..."
	@jq -r '.[] | select(.provider=="openai") | .models[]' $(CONFIG_FILE) | \
	xargs -I {} sh -c 'python3 src/run_eval_on_model.py --model openai/{} --eval_data_file $$EVAL_DATA_FILE --output "eval_outputs/output_openai_$$(basename {})_$$(basename $$EVAL_DATA_FILE)"'
