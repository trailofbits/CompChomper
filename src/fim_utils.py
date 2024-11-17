import argparse
import pandas as pd
import multiprocessing
import numpy as np
import math
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import itertools


def should_fim(fim_rate, rng):
    return rng.binomial(1, fim_rate) == 1

def should_spm(spm_rate, rng):
    return rng.binomial(1, spm_rate) == 1

def split_fim(txt, rng):
    boundaries = list(rng.randint(low=0, high=len(txt) + 1, size=2))
    boundaries.sort()

    prefix = txt[: boundaries[0]]
    middle = txt[boundaries[0] : boundaries[1]]
    suffix = txt[boundaries[1] :]

    return (prefix, middle, suffix)


def fim(data, fim_percent, template, seed, spm_percent, txt_mode):

    texts = [row["code"] for (_, row) in data.iterrows()]
    np_rng = np.random.RandomState(seed=seed)
    chunks = []

    if txt_mode:
        spm_func = template.spm_txt
        psm_func = template.psm_txt
    else:
        spm_func = template.spm_json
        psm_func = template.psm_json

    for text in texts:
        if should_fim(fim_percent, np_rng):
            (p, m, s) = split_fim(text, np_rng)
            if should_spm(spm_percent, np_rng):
                chunk = spm_func(pre_txt=p, mid_txt=m, suf_txt=s)
            else:
                chunk = psm_func(pre_txt=p, mid_txt=m, suf_txt=s)
            chunks.append(chunk)
        else:
            if txt_mode:
                chunks.append(
                    json.dumps(
                        {
                            "text": text,
                        }
                    )
                )
            else:
                chunks.append(
                    json.dumps(
                        {
                            "segments": [
                                {
                                    "label": True,
                                    # add eos and bos tokens as needed
                                    "text": template.nofim(text),
                                }
                            ]
                        }
                    )
                )

    return pd.DataFrame(chunks, columns=["json_fim"])


def load_data(input_file):
    if input_file.endswith(".feather"):
        df = pd.read_feather(input_file)
    elif input_file.endswith(".parquet"):
        df = pd.read_parquet(input_file)
    else:
        raise ValueError(
            "Unsupported file type. Only .feather and .parquet are supported."
        )
    return df


class FIM:
    def __init__(self, prefix_tok, middle_tok, suffix_tok, eos_tok):
        self.PRE = prefix_tok
        self.MID = middle_tok
        self.SUF = suffix_tok
        self.EOS = eos_tok

    def nofim(self, txt):
        if hasattr(self, "BOS"):
            txt = f"{self.BOS}{txt}{self.EOS}"
        else:
            txt = f"{txt}{self.EOS}"
        return txt

    def _spm(self, pre_txt, mid_txt, suf_txt):
        spm = [
            (True, f"{self.PRE}"),
            (True, f"{self.SUF}{suf_txt}"),
            (True, f"{self.MID}{pre_txt}"),
        ]

        # add eos when training
        # but do not add it if inferencing (aka no middle)
        if mid_txt:
            spm += [
                (True, f"{mid_txt}"),
            ]
            spm += [(True, f"{self.EOS}")]

        return spm

    def _psm(self, pre_txt, mid_txt, suf_txt):
        psm = [
            (True, f"{self.PRE}{pre_txt}"),
            (True, f"{self.SUF}{suf_txt}"),
            (True, f"{self.MID}"),
        ]

        # add eos when training
        # but do not add it if inferencing (aka no middle)
        if mid_txt:
            psm += [(True, f"{mid_txt}")]
            psm += [(True, f"{self.EOS}")]

        return psm

    def _to_txt(self, pre_txt, mid_txt, suf_txt, fn):
        lst = [y for (x, y) in fn(pre_txt, mid_txt, suf_txt)]
        _txt = "".join(lst)

        # completion mode will add bos and eos, remove it here
        if hasattr(self, "BOS") and _txt.startswith(self.BOS):
            _txt = _txt[len(self.BOS):]

        if _txt.endswith(self.EOS):
            _txt = _txt[:-1*len(self.EOS)]

        return json.dumps({"text": _txt})

    def psm_txt(self, pre_txt, mid_txt, suf_txt):
        return self._to_txt(pre_txt, mid_txt, suf_txt, self._psm)

    def spm_txt(self, pre_txt, mid_txt, suf_txt):
        return self._to_txt(pre_txt, mid_txt, suf_txt, self._spm)

    def _to_json(self, pre_txt, mid_txt, suf_txt, fn):
        tuples = fn(pre_txt, mid_txt, suf_txt)
        segs = []
        for x, y in tuples:
            seg = {"label": x, "text": y}
            segs.append(seg)

        dct = {"segments": segs}
        in_json = json.dumps(dct)
        return in_json

    def psm_json(self, pre_txt, mid_txt, suf_txt):
        return self._to_json(pre_txt, mid_txt, suf_txt, self._psm)

    def spm_json(self, pre_txt, mid_txt, suf_txt):
        return self._to_json(pre_txt, mid_txt, suf_txt, self._spm)


class DeepSeekFIM(FIM):
    def __init__(self):

        # DeepSeek uses "PMS" mode and not PSM mode, so we
        # switch S with M to make all the other code sane
        super().__init__(
            prefix_tok="<｜fim▁begin｜>",
            middle_tok="<｜fim▁end｜>",
            suffix_tok="<｜fim▁hole｜>",
            eos_tok="<｜end▁of▁sentence｜>",
        )
        self.BOS = "<｜begin▁of▁sentence｜>"

    def _psm(self, pre_txt, mid_txt, suf_txt):
        # taken from https://arxiv.org/pdf/2401.14196 page 8
        base_psm = super()._psm(pre_txt, mid_txt, suf_txt)
        bos = [
            (True, f"{self.BOS}"),
        ]

        base_psm = bos + base_psm

        return base_psm

    def _spm(self, pre_txt, mid_txt, suf_txt):
        # taken from https://arxiv.org/pdf/2401.14196 page 8
        base_spm = super()._spm(pre_txt, mid_txt, suf_txt)
        bos = [
            (True, f"{self.BOS}"),
        ]

        base_spm = bos + base_spm

        return base_spm


class CodeLlamaFIM(FIM):
    def __init__(self):
        super().__init__(
            prefix_tok="<PRE>", middle_tok="<MID>", suffix_tok="<SUF>", eos_tok="</s>"
        )
        self.BOS = "<s>"

    def _psm(self, pre_txt, mid_txt, suf_txt):
        # taken from https://github.com/meta-llama/codellama/blob/main/llama/generation.py#L495
        #
        base_psm = super()._psm(pre_txt, mid_txt, suf_txt)
        bos = [
            (True, f"{self.BOS}"),
        ]

        base_psm = bos + base_psm
        return base_psm

    def _spm(self, pre_txt, mid_txt, suf_txt):
        # taken from https://github.com/meta-llama/codellama/blob/main/llama/generation.py#L495
        #
        base_psm = super()._spm(pre_txt, mid_txt, suf_txt)
        bos = [
            (True, f"{self.BOS}"),
        ]

        base_psm = bos + base_psm
        return base_psm


class CodeQwenFIM(FIM):
    def __init__(self):
        # taken from https://github.com/QwenLM/CodeQwen1.5?tab=readme-ov-file#2-file-level-code-completion-fill-in-the-middle
        super().__init__(
            prefix_tok="<fim_prefix>",
            middle_tok="<fim_middle>",
            suffix_tok="<fim_suffix>",
            eos_tok="<|endoftext|>",
        )

        self.BOS = self.EOS

class QwenCoder25FIM(FIM):
    def __init__(self):
        # taken from https://github.com/QwenLM/Qwen2.5-Coder/blob/main/examples/Qwen2.5-Coder-fim.py
        super().__init__(
            prefix_tok="<|fim_prefix|>",
            middle_tok="<|fim_middle|>",
            suffix_tok="<|fim_suffix|>",
            eos_tok="<|endoftext|>",
        )

        self.BOS = self.EOS

class CodeGemmaFIM(FIM):
    def __init__(self):
        # taken from https://ai.google.dev/gemma/docs/formatting
        super().__init__(
            prefix_tok="<|fim_prefix|>",
            middle_tok="<|fim_middle|>",
            suffix_tok="<|fim_suffix|>",
            eos_tok="<eos>",
        )

        self.BOS = "<bos>"

class StarCoder2FIM(FIM):
    def __init__(self):
        super().__init__(
            prefix_tok="<fim_prefix>",
            middle_tok="<fim_middle>",
            suffix_tok="<fim_suffix>",
            eos_tok="<|endoftext|>",
        )

    # def _psm(self, pre_txt, mid_txt, suf_txt):
    #    # the Starcoder2 paper says that it was trained on this full format for FIM
    #    # taken from https://arxiv.org/pdf/2402.19173 page 16
    #    # psm_txt = "<repo_name>reponame<file_sep>{self.PRE}filepath0\n{pre_txt}{self.SUF}{suf_txt}{self.MID}{mid_txt}{self.EOS}"
    #    # psm = [
    #    #    #(False, "<repo_name>reponame"),
    #    #    #(False, "<file_sep>"),
    #    #    (True, f"{self.PRE}"),
    #    #    #(False, "filepath0"),
    #    #    #(False, "\n"),
    #    #    (True, f"{pre_txt}"),
    #    #    (True, f"{self.SUF}{suf_txt}"),
    #    #    (True, f"{self.MID}{mid_txt}"),
    #    #    #(False, f"{self.EOS}"),
    #    # ]
    #    psm = super()._psm(pre_txt, mid_txt, suf_txt)
    #    return psm

class CodestralFIM(FIM):
    def __init__(self):
        super().__init__(
            prefix_tok="[PREFIX]",
            # codestral has no middle token, leave blank
            middle_tok="",
            suffix_tok="[SUFFIX]>",
            eos_tok="</s>",
        )

def load_template_by_model(model):
    provider, model_id = model.split("/", maxsplit=1)
    if provider == "ollama":
        model_id, _ = model_id.split(":", maxsplit=1)
    else:
        model_id, _ = model_id.split("-", maxsplit=1)

    model_id = model_id.lower()

    supported = {
        "starcoder2": StarCoder2FIM(),
        # both are codeqwen1.5
        "codeqwen": CodeQwenFIM(),
        "codeqwen1.5": CodeQwenFIM(),
        "qwen2.5-coder": QwenCoder25FIM(),
        "qwen2.5": QwenCoder25FIM(),
        "codellama": CodeLlamaFIM(),
        "codegemma": CodeGemmaFIM(),
        # both are deepseek coder
        "deepseek": DeepSeekFIM(),
        "deepseek-coder": DeepSeekFIM(),
        "deepseek-coder-v2": DeepSeekFIM(),
        "codestral": CodestralFIM(),
    }

    template = supported.get(model_id, None)
    if template is None:
        raise RuntimeError(f"Can't handle model of type: {model} [{model_id}]")
    else:
        print(f"Using template for {model_id}")

    return template


def main(args):
    df = load_data(args.input_file)
    fim_percent = args.fim_percent
    assert 0.0 <= fim_percent <= 1.0
    spm_percent = args.spm_percent
    assert 0.0 <= spm_percent <= 1.0
    psm_percent = 1.0 - spm_percent

    print(f"Running at fim {fim_percent} split {spm_percent} spm / {psm_percent} psm")
    # print(f"Running at fim {fim_percent} in psm mode [seed={args.seed}]")

    model_template = load_template_by_model(args.model)

    num_cores = multiprocessing.cpu_count()
    # Splitting the dataframe into nearly equal parts
    split_dfs = np.array_split(df, num_cores)

    results = []
    counter = itertools.count()
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Map each part of the DataFrame to the chunk function
        futures = {}
        for split_df in split_dfs:
            idx = next(counter)
            futures[idx] = executor.submit(
                fim,
                split_df,
                fim_percent,
                model_template,
                int(args.seed),
                spm_percent,
                args.txt_mode,
            )

        for _, future in futures.items():
            result = future.result()
            results.append(result)

    # Concatenate all results into a new DataFrame
    new_df = pd.concat(results, ignore_index=True)

    print(f"Saving results to {args.output_file}")
    with open(args.output_file, "w") as output_json:
        for _, row in new_df.iterrows():
            output_json.write(row["json_fim"])
            output_json.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and chunk data files.")
    parser.add_argument(
        "--fim_percent",
        type=float,
        default=0.80,
        help="percentage [0, 1.0] of fim to apply",
    )

    parser.add_argument(
        "--spm_percent",
        type=float,
        default=0.50,
        help="percentage [0, 1.0] of fim to do in spm (rest will be psm)",
    )

    parser.add_argument(
        "--txt_mode",
        action="store_true",
        help="use completion mode and not segment mode",
    )
    parser.add_argument(
        "--model", type=str, required=True, help="which model to psm for"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("input_file", type=str, help="Input file path.")
    parser.add_argument("output_file", type=str, help="Output file path.")
    args = parser.parse_args()

    main(args)
