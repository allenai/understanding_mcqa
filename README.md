This repository contains code for the arXiv preprint ["Answer, Assemble, Ace: Understanding How Transformers Answer Multiple Choice Questions"](https://arxiv.org/abs/2407.15018).

I've cleaned this code up substantially from the version used to run the experiments in the paper, so if you have any replication issues, please open an issue or email me!

If you use this code or find our paper valuable, please cite:
```
@unpublished{wiegreffe2024answer,
  title={Answer, Assemble, Ace: Understanding How Transformers Answer Multiple Choice Questions},
  author={Wiegreffe, Sarah and Tafjord, Oyvind, and Belinkov, Yonatan and Hajishirzi, Hannaneh and Sabharwal, Ashish},
  year={2024},
  note={arXiv:2107.15018},
  url={https://arxiv.org/abs/2407.15018}
}
```

Some of this code is inspired by or modified from Kevin Meng's [memit codebase](https://github.com/kmeng01/memit) and Jack Merullo's [lm_vector_arithmetic codebase](https://github.com/jmerullo/lm_vector_arithmetic), and I have mentioned this explicitly in the docstrings of the relevant files.

The Memory Colors dataset (`memory_colors.csv`) is extracted from [this paper's](https://aclanthology.org/2021.blackboxnlp-1.10.pdf) Appendix table.

## Setup

Necessary packages for the project are listed in `requirements.txt`.
Code is formatted with `black` and `isort`.

You'll also need to clone the [MEMIT repository](https://github.com/kmeng01/memit) as we will use their file `util/nethook.py` to capture hidden states in the model. Once you clone the directory, you can either 1) copy the file into a folder called `util/` in this directory, or 2) add the directory to your `PYTHONPATH` (such as via `export PYTHONPATH=$PYTHONPATH:/path/to/memit/`). The script is called as `from util import nethook`.


## Usage

The main entrypoint is the `run_experiments.py` script. For each run, you will need to specify a `--model` and `--dataset`.

A standard command looks as follows:
```
python run_experiments.py
  --in_context_examples 3
  --icl_labels ABB
  --model {llama2-7b, llama2-13b, llama2-7b-chat, llama2-13b-chat, olmo-7b, olmo-7b-sft, olmo-v1.7-7b, olmo-v1.7-7b-350B, olmo-v1.7-7b-1T}
  --llama_path /path/to/llama2 [required only if using llama2 models]
  --dataset {hellaswag, mmlu, prototypical_colors}
```
For HellaSwag, add `--split val` to replicate the paper in running on the validation set.

For the "A:\_\_**B:**\_\_" prompt format instead of "**A:**\_\_B:\_\_", add `--inverse`.

For the "**C:**\_\_D:\_\_" prompt format, add `--cd_label_mapping`.

For "C:\_\_**D:**\_\_", add both `--inverse` and `--cd_label_mapping`.

All results will write out to a `results/` directory in the root of the project. Add flag `--override` to overwrite existing results.

### Running Evaluation of Models
Add `--function calculate_acc`.

### Running Vocabulary Projection
Add
```
--function {vocab_projection_coarse, vocab_projection_finegrained_mlpOutput, vocab_projection_finegrained_attnOutput, vocab_projection_finegrained_attnHeads}
```

### Running Activation Patching
Add
```
  --function {trace_all, trace_layer, trace_attn, trace_mlp, trace_attn_heads}
  --base_prompt {standard, inverse, cd_label_mapping, cd_label_mapping+inverse}
  --prompt_to_sub_in {standard, inverse, cd_label_mapping, cd_label_mapping+inverse}
  --num_instances 1000 [if directly replicating the paper]
```

### Non-MCQA Experiments
For experiments on the generative version of the Colors task in section 4.3 & Figure 3 of the paper, add `--scoring_type next_token`. Otherwise, by default, the scripts will use `--scoring_type enumerated`, meaning that the prompt will be formatted as MCQA and model performance will be judged by comparing the scores assigned to the "A" and "B" tokens.
