# Solidity Technical Evaluation Details
Writing a good evaluation is very difficult and writing a perfect one is impossible.

Here are some of the specific tradeoffs we made when crafting the Solidity completion benchmark and why we chose them.

* Whitespace matters in our benchmark because we think code completion should maintain code flow and spacing.
* Tokenization provides a source of variability between models, especially when measuring things in tokens. To maintain one single source of test data, we do not split it along token boundaries but via "faketokens", where by default 1 faketoken = 2 characters. This is adjustable via command-line arguments. See [generate_eval_from_raw_code.py](../src/generate_eval_from_raw_code.py) for details.
* The context window is by default 1024 (real) tokens and we generate 128 (real) tokens of output. This was a compromise between latency and providing enough context.
* We do not try to trick the model with zero character completions.
* We include the comments and Solidity pragmas in the benchmark. A good code model should also complete comments for code and not only code itself.

# Writing Evaluation Prompts

To achieve accurate evaluations, we start by examining the tokens which AI coding assistants actually send to the model for completion. This is the critical data we need to capture, as [subtle changes to token order or tokenization gotchas can significantly degrade model performance](https://hamel.dev/notes/llm/finetuning/05_tokenizer_gotchas.html).

In practical terms, where do we find what tokens code models see? For Tabby, the relevant information is located in the `~/.tabby/events/<date>.json` file, as detailed in [Tabby's documentation](https://tabby.tabbyml.com/docs/administration/context/). For Continue, the output can be viewed directly in the output tab of VS Code.

When manually examining tokens these tools send, an important detail emerges: the task we are dealing with is called ["Fill In the Middle" (FIM)](https://arxiv.org/abs/2207.14255), which uses special tokens to designate the prefix (start of code), middle (the part to fill), and suffix (the end it connects to). Understanding this is crucial, as our evaluations will focus on how different models handle Solidity FIM, specifically in "PSM" (prefix-suffix-middle) mode, which is commonly used by code editors.

Once we know *which* tokens are sent, its time to figure out how many. In terms of prompt size and generated token count, there are notable differences between tools. Tabby uses a [context size of 1024 tokens with 512 reserved for the prompt, and generates 64 tokens](https://github.com/TabbyML/tabby/blob/989c2ababdd7d936c22749bf1e6b096b4ede2b9f/crates/tabby/src/services/completion.rs#L268). On the other hand, Continue.dev uses a [500-token prompt and generates 1024 tokens](https://github.com/continuedev/continue/blob/6c6be05dbf8e54f4a4c196bf8a76328799c6b5ad/core/util/parameters.ts#L7).

For the purposes of our evaluation, we will standardize on a 1024-token context and 128 generated tokens as a reasonable compromise between the two approaches.

# Avoid Evaluating on the Training Set

One of the easiest ways to win an LLM benchmark is to train on evaluation data. We try to minimize accidentally evaluating the training set by evaluating on code ostensibly written after the training cutoff date. As new code is often based on (or directly re-uses) old code, this may not be a panacea.

A great source of such Solidity code are contracts found in the [code-423n4 Github repository](https://github.com/code-423n4). Their examples are particularly well-suited for our purposes because they are dated, continually updated, and are from "real" Solidity code in need of audits.

# Tokenizing & Scoring

After gathering the raw Solidity code, the next step is to transform the code into a large dataset of expected completions. Given that we're dealing with multiple different models, it is important to format and tokenize the data for each one.

The evaluation process itself can be broken down into the following steps:

Select Completions: Select the number of completions you'd want from each file (N). Then the algorithm to generate copmletions looks like the following pseudocode.

```
let N be the number of completions desired from each file
for file in all_files:
    content = file.content()
    for i in range(N):
        prefix, middle, suffix = split_data(content)
        completions.append((prefix, milddle, suffix))
```

Scoring: Each completion is scored on a scale from 0 to 1, based on its accuracy.
Averaging Scores: Calculate the average score for each file.
Total Score Calculation: Sum the average scores from all files to obtain the total score for the model. This method is designed to prevent a model from being disproportionately penalized for specific faults while maintaining a fair overall assessment.

The large commercial models don’t expose a FIM interface over API, so we used prompts borrowed from Continue.dev to simulate code infill capability.


* How do we grade similarity? This is a tough question! Imagine the expected value of a completion is foo(bar). There are two models, A and B. Model A answers foo( bar ) and model B answer frob(bar). These answers are equally distant (two edits apart), but model A is clearly better. We used three measurements BLEU, CodeBLEU and Insert/Delete (Indel) Distance as possible metrics. We present the Indel scores as these felt the most useful and consistent.