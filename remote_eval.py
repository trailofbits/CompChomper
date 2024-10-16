#heavily borrowed from https://github.com/modal-labs/llm-finetuning/tree/main/src
import os
from pathlib import PurePosixPath
from typing import Union
import modal

APP_NAME = "CompChomper-LLM-eval"

MINUTES = 60  # seconds
HOURS = 60 * MINUTES


def get_cuda_image(base="nvidia/cuda:12.1.0-base-ubuntu22.04"):
    my_reqs_txt = os.path.join(os.path.dirname(__file__), "requirements.txt")
    with open(my_reqs_txt, 'r') as reqs_file: 
        my_reqs = [l.strip() for l in reqs_file.readlines()]
        print(f"reqs: {my_reqs}")

    apt_commands = [
        "jq", "make", "curl", "git", "pciutils", "cuda-toolkit"
    ]

    img = modal.Image.from_registry(base, add_python="3.10")
    img = img.apt_install(*apt_commands)
    img = img.run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    img = img.pip_install(*my_reqs).run_commands("python3 -m pip install flash-attn==2.6.1 --no-build-isolation")
    # for some reason extra_options to pip doesn't work
    img = img.env({"OLLAMA_MODELS": "/app/eval_models/ollama"})
    img = img.env({"HF_HOME": "/app/eval_models/hf"})

    return img

app = modal.App(APP_NAME)

# Volumes for pre-trained models and training runs.
eval_packages = modal.Volume.from_name(
    "eval_packages", create_if_missing=True
)
eval_outputs = modal.Volume.from_name(
    "eval_outputs", create_if_missing=True
)
eval_models = modal.Volume.from_name(
    "eval_models", create_if_missing=True
)

eval_raw_files = modal.Volume.from_name(
    "eval_raw_files", create_if_missing=True
)
VOLUME_CONFIG: dict[Union[str, PurePosixPath], modal.Volume] = {
    "/app/eval_packages"   : eval_packages,
    "/app/eval_outputs"    : eval_outputs,
    "/app/eval_models"     : eval_models,
    "/app/eval_raw_files"  : eval_raw_files,
}

MOUNT_CONFIG = [
    modal.Mount.from_local_dir("./src", remote_path="/app/src"),
    modal.Mount.from_local_dir("./config", remote_path="/app/config"),
    modal.Mount.from_local_file("./Makefile", remote_path="/app/Makefile")
]

#SINGLE_GPU_CONFIG = os.environ.get("GPU_CONFIG", "a10g:1")
# we need more vram for larger models
SINGLE_GPU_CONFIG = os.environ.get("GPU_CONFIG", "a100-80gb:1")


def get_commands(provider, packages):
    base_command = ""
    commands = []
    
    if provider == "ollama":
        base_command  = "(ollama serve)& sleep 4"
        base_command += " && mkdir -p ${OLLAMA_MODELS}"
        base_command += " && cd /app"
        base_command += " && make fetch_ollama"

        for package in packages:
            run_command = base_command
            run_command += f" && make eval_ollama EVAL_DATA_FILE={package}"
            commands.append(run_command)
    elif provider == "hf":
        base_command =  "cd /app && make fetch_hf"

        for package in packages:
            run_command = base_command
            run_command += f" && make eval_hf EVAL_DATA_FILE={package}"
            commands.append(run_command)
    elif provider == "openai":
        base_command =  "cd /app && make fetch_openai"

        for package in packages:
            run_command = base_command
            run_command += f" && make eval_openai EVAL_DATA_FILE={package}"
            commands.append(run_command)
    elif provider == "claude":
        base_command =  "cd /app && make fetch_claude"

        for package in packages:
            run_command = base_command
            run_command += f" && make eval_claude EVAL_DATA_FILE={package}"
            commands.append(run_command)

    return commands

def generic_eval_runner(provider):

    packages = os.environ.get("EVAL_DATA_FILE", None)
    if packages is None:
        raise RuntimeError("Please specify a list of packages to eval in EVAL_DATA_FILE env var")

    packages = packages.split(";")
    commands = get_commands(provider, packages)
    for command in commands:
        run_in_sandbox(command)

@app.local_entrypoint()
def eval_ollama():
    generic_eval_runner("ollama")

@app.local_entrypoint()
def eval_hf():
    generic_eval_runner("hf")

@app.local_entrypoint()
def eval_openai():
    generic_eval_runner("openai")

@app.local_entrypoint()
def eval_claude():
    generic_eval_runner("claude")

@app.local_entrypoint()
def make_eval_packages():
    run_in_sandbox("cd /app && make eval_packages_2024-06")

@app.local_entrypoint()
def fetch_all():
    run_in_sandbox("(ollama serve)& sleep 4 && cd /app && make fetch_all")

@app.local_entrypoint()
def main():
    fetch_all()
    make_eval_packages()
    eval_openai()
    eval_claude()
    eval_hf()
    eval_ollama()

def run_in_sandbox(cmd: str):
    sb = app.spawn_sandbox(
        "bash",
        "-c",
        cmd,
        image=get_cuda_image(),
        mounts=MOUNT_CONFIG,
        volumes=VOLUME_CONFIG,
        timeout=24 * HOURS,
        gpu=SINGLE_GPU_CONFIG,
        _allow_background_volume_commits=True,
        secrets=[modal.Secret.from_dotenv()],

    )

    sb.wait()

    if sb.returncode != 0:
        print(f"Run failed with code {sb.returncode}")
        print(sb.stderr.read())
    else:
        print(f"Run success")
        print(sb.stdout.read())
