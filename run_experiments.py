import argparse
import csv
import os
import random
import time

import warnings
from collections import defaultdict

import datasets
import pandas as pd
import torch
from tqdm import tqdm

from util.scoring_and_patching_utils import *
from util.vocab_projection_utils import *

# matplotlib produces a lot of warnings-- silence them
warnings.filterwarnings("ignore")

torch.set_grad_enabled(False)

mmlu_tasks = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


def format_datapoints(
    knowns,
    type,
    model,
    in_context_examples=None,
    scoring_type=None,
):
    formatted_datapoints = []
    formatted_icl = []

    if type == "hellaswag":
        for i, item in tqdm(enumerate(knowns), total=len(knowns)):
            # get labels
            label_1 = item["endings"][int(item["label"])]  # correct
            remaining_options = [el for el in range(len(knowns[0]["endings"]))]
            remaining_options.remove(int(item["label"]))
            random.seed(10)
            negative_index = random.choice(remaining_options)
            label_2 = item["endings"][negative_index]  # incorrect

            pair = {}
            pair["prompt"] = item["ctx"]
            pair["completion_one"] = (label_1, "")
            pair["completion_two"] = (label_2, "")

            formatted_datapoints.append(pair)

        if in_context_examples is not None:
            for item in in_context_examples:
                # get labels
                label_1 = item["endings"][int(item["label"])]  # correct
                remaining_options = [el for el in range(len(knowns[0]["endings"]))]
                remaining_options.remove(int(item["label"]))
                random.seed(10)
                label_2 = item["endings"][random.choice(remaining_options)]  # incorrect

                pair = {}
                pair["prompt"] = item["ctx"]
                pair["completion_one"] = (label_1, "")
                pair["completion_two"] = (label_2, "")
                formatted_icl.append(pair)

    elif type == "prototypical_colors":
        if "llama2" in model:
            all_colors = list(set([el.strip() for el in knowns[" Color"].unique().tolist()]))
        elif "olmo" in model:
            all_colors = list(set([" " + el.strip() for el in knowns[" Color"].unique().tolist()]))

        for i, item in knowns.iterrows():
            # get labels
            label_1 = item[" Color"].strip()  # correct
            all_colors = [
                "black",
                "orange",
                "brown",
                "white",
                "blue",
                "pink",
                "grey",
                "purple",
                "red",
                "yellow",
                "green",
            ]
            all_colors.remove(label_1)
            random.seed(10)
            if scoring_type == "next_token":
                # return all other possible colors to search through
                label_2 = all_colors
            elif scoring_type == "enumerated":
                # select a negative example
                label_2 = random.choice(all_colors)
            else:
                breakpoint()

            pair = {}
            verb = "is" if isinstance(item["Plural"], float) else "are"
            noun = f"{item[' Descriptor'].strip() if not isinstance(item[' Descriptor'], float) else ''} {item[' Item'].strip()}".strip()
            pair["prompt"] = (
                f"{noun} {verb} {label_1}.".capitalize() + f" What color {verb} {noun}?"
            )
            pair["completion_one"] = (label_1, "")
            pair["completion_two"] = (label_2, "")

            formatted_datapoints.append(pair)

    elif type == "mmlu":
        # need to store few-shot examples for each sub-task for easy indexing
        formatted_icl = {}
        i = 0
        for subtask_name, sub_dataset in tqdm(knowns.items(), total=len(knowns)):
            # process main examples
            for item in sub_dataset["test"]:
                # get labels
                label_1 = item["choices"][int(item["answer"])]  # correct
                remaining_options = [el for el in range(4)]
                remaining_options.remove(int(item["answer"]))
                random.seed(10)
                negative_index = random.choice(remaining_options)
                label_2 = item["choices"][negative_index]  # incorrect

                # odd instance where answer choices are the same
                if label_1 == label_2:
                    # cycle through the other options and see if any others match the correct answer choice in tokenized length
                    remaining_options.remove(negative_index)
                    random.shuffle(remaining_options)
                    for remaining_idx in remaining_options:
                        label_2 = item["choices"][remaining_idx]
                        if label_1 != label_2:
                            break

                pair = {}
                pair["prompt"] = item["question"]
                pair["completion_one"] = (label_1, "")
                pair["completion_two"] = (label_2, "")
                pair["task"] = subtask_name

                if label_1 != label_2:
                    formatted_datapoints.append(pair)
                    i += 1
            if in_context_examples is not None:
                # there are 5 in-context examples per sub-task
                tmp_icl_examples = []
                for item in sub_dataset["dev"]:
                    # get labels
                    label_1 = item["choices"][int(item["answer"])]  # correct
                    remaining_options = [el for el in range(4)]
                    remaining_options.remove(int(item["answer"]))
                    random.seed(10)
                    label_2 = item["choices"][random.choice(remaining_options)]  # incorrect

                    pair = {}
                    pair["prompt"] = item["question"]
                    pair["completion_one"] = (label_1, "")
                    pair["completion_two"] = (label_2, "")
                    tmp_icl_examples.append(pair)
                # select the specified # of examples
                random.seed(10)
                tmp_icl_examples = random.sample(tmp_icl_examples, in_context_examples)
                formatted_icl[subtask_name] = tmp_icl_examples
    else:
        raise Exception("invalid dataset type")

    print(f"searched through {i+1} points.")

    return formatted_datapoints, formatted_icl


if __name__ == "__main__":
    og_start_time = time.time()

    next_token_functions = [
        "calculate_acc_next_token",
        "vocab_projection_coarse_next_token",
        "vocab_projection_finegrained_mlpOutput_next_token",
        "vocab_projection_finegrained_attnOutput_next_token",
        "vocab_projection_finegrained_attnHeads_next_token",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llama_path", type=str, default=None, help="path to llama2 model checkpoints"
    )
    parser.add_argument(
        "--include_negatives",
        action="store_true",
        help="use if want to run tracing for incorrectly predicted examples as well",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "llama2-7b",
            "llama2-13b",
            "llama2-7b-chat",
            "llama2-13b-chat",
            "olmo-7b",
            "olmo-7b-sft",
            "olmo-v1.7-7b",
            "olmo-v1.7-7b-350B",
            "olmo-v1.7-7b-1T",
        ],
    )
    parser.add_argument(
        "--scoring_type",
        default="enumerated",
        choices=[
            "enumerated",
            "next_token",
        ],
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "hellaswag",
            "prototypical_colors",
            "mmlu",
        ],
    )
    parser.add_argument(
        "--function",
        required=True,
        choices=[
            "trace_all",
            "trace_layer",
            "trace_mlp",
            "trace_attn",
            "trace_attn_heads",
            "calculate_acc",
            "vocab_projection_coarse",
            "vocab_projection_finegrained_mlpOutput",
            "vocab_projection_finegrained_attnOutput",
            "vocab_projection_finegrained_attnHeads",
        ]
        + next_token_functions,
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="top k tokens to save when performing vocabulary projection",
    )
    parser.add_argument(
        "--base_prompt",
        required=False,
        default=None,
        type=str,
        choices=[
            "standard",
            "inverse",
            "cd_label_mapping",
            "cd_label_mapping+inverse",
        ],
    )
    parser.add_argument(
        "--prompt_to_sub_in",
        required=False,
        default=None,
        type=str,
        choices=[
            "standard",
            "inverse",
            "cd_label_mapping",
            "cd_label_mapping+inverse",
        ],
    )
    parser.add_argument("--in_context_examples", type=int, default=0)
    parser.add_argument("--icl_labels", type=str, choices=["A", "B", "AB", "BA", "ABB", "BAA"]),
    parser.add_argument("--inverse", action="store_true")
    parser.add_argument("--cd_label_mapping", action="store_true", default=False)
    parser.add_argument("--split", default="test", choices=["train", "test", "validation"])
    parser.add_argument("--num_instances", default=5000, type=int)
    parser.add_argument(
        "--override",
        help="use this to allow overwrites of output files",
        action="store_true",
    )
    parser.add_argument(
        "--no_model_load",
        action="store_true",
        help="use to speed up debugging by not loading the model checkpoint",
    )
    args = parser.parse_args()

    if args.model in {"olmo-v1.7-7b-350B", "olmo-v1.7-7b-1T"}:
        raise Exception(
            "Intermediate Olmo model checkpoints not yet available on HF (https://huggingface.co/allenai/OLMo-7B-0424)"
        )

    if args.scoring_type == "next_token" and "next_token" not in args.function:
        args.function = args.function + "_next_token"
        assert args.function in next_token_functions
    if args.scoring_type == "next_token" and (
        args.dataset != "prototypical_colors" or args.inverse or args.cd_label_mapping
    ):
        raise Exception(
            "Next token sequence scoring doesn't support other tasks or multiple-choice specifications currently."
        )

    if "trace" in args.function:
        # must specify at least one alternative prompt from original where the correct answer choice's letter changes
        assert args.base_prompt is not None
        assert args.prompt_to_sub_in is not None
        assert args.base_prompt != args.prompt_to_sub_in

    if "llama" in args.model:
        if args.llama_path is None:
            raise Exception(
                "Must specify a path from which to load Huggingface llama model checkpoints"
            )

    print(
        "Running ",
        args.model,
        ": inverse=",
        args.inverse,
        ", cd_label_mapping=",
        args.cd_label_mapping,
    )

    kind = None
    if args.function == "trace_mlp":
        kind = "mlp"
    elif args.function == "trace_attn":
        kind = "attn"
    elif args.function == "trace_attn_heads":
        kind = "attn_heads"

    if args.in_context_examples > 0 and not args.icl_labels:
        # create a random ICL label ordering
        random.seed(10)
        args.icl_labels = "".join(
            [random.choice(("A", "B")) for _ in range(args.in_context_examples)]
        )
    if args.icl_labels is not None and (args.in_context_examples != len(args.icl_labels)):
        print(
            f"Warning: you specified {len(args.icl_labels)} in-context labels but {args.in_context_examples} in-context examples"
        )

    if not torch.cuda.is_available() and not args.no_model_load:
        raise Exception("Change runtime type to include a GPU.")

    in_context_examples = None

    if args.dataset == "hellaswag":
        # load Hellaswag dataset
        knowns = datasets.load_dataset("hellaswag", split="validation")
        if args.in_context_examples > 0:
            # load in-context examples
            in_context_examples = datasets.load_dataset(
                "hellaswag", split=f"train[:{args.in_context_examples}]"
            )
    elif args.dataset == "prototypical_colors":
        knowns = pd.read_csv("./data/memory_colors.csv")
    elif args.dataset == "mmlu":
        knowns = {}
        # iterate through all sub-tasks (57)
        for subtask_idx, subtask in tqdm(enumerate(mmlu_tasks), total=len(mmlu_tasks)):
            # load sub-dataset (dev, validation & test splits)
            knowns[subtask] = datasets.load_dataset("cais/mmlu", subtask)
        # specify how many ICL examples to take
        in_context_examples = args.in_context_examples

    # just load tokenizer until determine if will be running script fully or not
    mt = ModelAndTokenizer(args.model, no_model_load=True, llama_path=args.llama_path)

    # format
    if knowns is not None:
        og_len = (
            len(knowns)
            if args.dataset != "mmlu"
            else sum([len(knowns[task]["test"]) for task in knowns])
        )
        knowns, icl = format_datapoints(
            knowns,
            model=args.model,
            type=args.dataset,
            in_context_examples=in_context_examples,
            scoring_type=args.scoring_type,
        )
        if args.dataset in {"mmlu", "hellaswag"} and len(knowns) > args.num_instances:
            print(f"subsetting down dataset to {args.num_instances} datapoints")
            random.seed(10)
            # randomly select 5k elements from the list
            knowns = random.sample(knowns, args.num_instances)
        elif len(knowns) > args.num_instances:
            raise Exception("too many datapoints:", len(knowns))
        if args.dataset == "prototypical_colors":
            # pull random instances from knowns to be in-context examples
            if args.in_context_examples > 3:
                raise Exception(
                    "too many in-context examples-- will result in assessing perf on a smaller test set"
                )
            random.seed(10)
            random.shuffle(knowns)
            # hard-code the test set as 3:108 so same regardless of # ICL
            icl = knowns[: args.in_context_examples]
            knowns = knowns[3:]
        print(f"Using {len(knowns)} valid datapoints out of {og_len}")
        n_icl = len(icl) if args.dataset != "mmlu" else [len(icl[task]) for task in icl][0]
        print(f"Total ICL examples = {n_icl}")
        if n_icl != args.in_context_examples:
            raise Exception("didn't find enough in-context examples")

    records_filepath = f"./results/memit_outputs/{args.dataset}/{args.model}/files/val/"
    if "vocab_projection_finegrained_attnHeads" in args.function:
        records_filepath = records_filepath + "attn_head_vocab_projections/"
        extra_file_folder = ""
        if args.cd_label_mapping:
            extra_file_folder += "_CD"
        if args.inverse:
            extra_file_folder += "_inverse"
        records_filepath = records_filepath + extra_file_folder.strip("_") + "/"

    if not os.path.exists(records_filepath):
        os.makedirs(records_filepath)

    shared_substring = f"_{args.in_context_examples}ICE_{args.icl_labels}labels{'_inverse' if args.inverse else ''}{'_CD' if args.cd_label_mapping else ''}_{len(knowns)}insts"
    if "vocab_projection_finegrained_attnHeads" in args.function:
        records_subfile = f"{args.function}_top{args.k}_all{shared_substring}_layer.jsonl"
    elif "vocab_projection" in args.function:
        records_subfile = f"{args.function}_top{args.k}_all{shared_substring}.jsonl"
    elif "trace" in args.function:
        records_subfile = f"causalTrace_stateSwapping_results_{args.function.split('trace_')[1]}_{args.base_prompt}_to_{args.prompt_to_sub_in}{shared_substring}.jsonl"
    else:
        records_subfile = "n/a"
    records_filename = os.path.join(records_filepath, records_subfile)
    # check if outfile is empty or not
    if "calculate_acc" in args.function:
        acc_file = os.path.join(
            records_filepath,
            f"{args.function.split('calculate_')[1]}{shared_substring}.csv",
        )
        print("WRITING TO", acc_file)
        already_completed = 0
        if not args.override:
            if os.path.exists(acc_file):
                # get existing num lines
                with open(acc_file, "r") as temp_f:
                    reader = csv.DictReader(temp_f)
                    already_completed = sum(1 for _ in reader)
                if already_completed > len(knowns):
                    raise Exception(
                        "Issue: file longer than specified length: ",
                        already_completed,
                        "vs.",
                        len(knowns),
                    )
                elif already_completed == len(knowns):
                    raise Exception("Complete Acc file already exists")
                else:
                    print(f"Completing {len(knowns)-already_completed} insts")

    elif not args.override:
        already_completed = 0
        if os.path.exists(records_filename):
            already_completed = sum(1 for _ in open(records_filename))
            if already_completed > len(knowns):
                raise Exception(
                    "Issue: file longer than specified length: ",
                    already_completed,
                    "vs.",
                    len(knowns),
                )
            elif already_completed == len(knowns):
                raise Exception("Complete file already exists")
            else:
                print(f"Completing {len(knowns)-already_completed} insts")
        if args.function == "trace_all":
            # also check MLP and Attn
            if os.path.exists(records_filename.replace("_all", "_mlp")):
                mlp_already_completed = sum(
                    1 for _ in open(records_filename.replace("_all", "_mlp"))
                )
            if os.path.exists(records_filename.replace("_all", "_attn")):
                attn_already_completed = sum(
                    1 for _ in open(records_filename.replace("_all", "_attn"))
                )
            if not (attn_already_completed == mlp_already_completed == already_completed):
                raise Exception("difference between file lengths, and not all files are complete")
    else:
        already_completed = 0
    print("WRITING TO", records_filename)

    if not args.override and "vocab_projection_finegrained_attnHeads" in args.function:
        # check that individual files don't yet exist
        completed_per_layer = defaultdict(int)
        completed = 0
        for layer_idx in range(mt.num_layers):
            layerwise_filename = records_filename.replace("layer", f"layer{layer_idx}")
            if os.path.exists(layerwise_filename):
                lw_already_completed = sum(1 for _ in open(layerwise_filename))
                if lw_already_completed > len(knowns):
                    raise Exception(
                        "Issue: file longer than specified length: ",
                        lw_already_completed,
                        "vs.",
                        len(knowns),
                    )
                elif lw_already_completed == len(knowns):
                    print(f"Complete layerwise file {layer_idx} already exists")
                    completed += 1
                else:
                    print(
                        f"Completing {len(knowns)-lw_already_completed} insts in layerwise file {layer_idx}"
                    )
                    completed_per_layer[layer_idx] = lw_already_completed
        if completed == mt.num_layers:
            raise Exception("All layerwise files have already been completed")
        else:
            # find the min instance value to start at
            already_completed = min([completed_per_layer[k] for k in range(mt.num_layers)])

    times, corrs, ind_corrs, inv_corrs, inv_ind_corrs = [], [], [], [], []
    cons_correct, cons_incorr, incons_A, incons_B = [], [], [], []
    traced = 0

    # now, load model
    mt = ModelAndTokenizer(args.model, no_model_load=False, llama_path=args.llama_path)
    # and setup model wrapper
    if not args.no_model_load:
        if "llama" in args.model:
            wrapper = LLaMAWrapper(mt.model, mt.tokenizer)
        elif "olmo" in args.model:
            wrapper = OlmoWrapper(mt.model, mt.tokenizer)
        else:
            raise Exception("model wrapper not yet written for this model")

    if "vocab_projection" in args.function:
        if "llama2" in args.model:
            lm_head = wrapper.model.lm_head.weight
        elif "olmo" in args.model:
            # 1B model has weight tying & thus uses wte instead of ff_out for unembedding
            assert "7b" in args.model
            lm_head = wrapper.model.model.transformer.ff_out.weight
            # bias is only used (throughout all linear layers of model) if the following flag is true
            if wrapper.model.model.config.include_bias:
                raise Exception(
                    "Model (unexpectedly) includes bias terms, and the projection code does not currently account for this."
                )
        else:
            raise Exception("model not supported for vocab projection")

    # also get token indices
    if args.scoring_type == "enumerated":
        if "llama2" in args.model:
            if args.cd_label_mapping:
                tok_1 = "C"
                tok_2 = "D"
                idx_1 = 315
                idx_2 = 360
                real_tok_1 = "A"
                real_tok_2 = "B"
                real_idx_1 = 319
                real_idx_2 = 350
            else:
                tok_1 = "A"
                tok_2 = "B"
                idx_1 = 319
                idx_2 = 350
        elif "olmo" in args.model:
            # shared by v1 and v1.7 models
            if args.cd_label_mapping:
                tok_1 = " C"
                tok_2 = " D"
                idx_1 = 330
                idx_2 = 399
                real_tok_1 = " A"
                real_tok_2 = " B"
                real_idx_1 = 329
                real_idx_2 = 378
            else:
                tok_1 = " A"
                tok_2 = " B"
                idx_1 = 329
                idx_2 = 378
        else:
            raise Exception("model not supported for vocab projection")
        a_token_index = wrapper.tokenizer.encode(tok_1)[-1]
        b_token_index = wrapper.tokenizer.encode(tok_2)[-1]
        if a_token_index != idx_1 or b_token_index != idx_2:
            raise Exception("tokenization issue: check answer choice tokens!")
        if args.cd_label_mapping:
            real_a_token_index = wrapper.tokenizer.encode(real_tok_1)[-1]
            real_b_token_index = wrapper.tokenizer.encode(real_tok_2)[-1]
            if real_a_token_index != real_idx_1 or real_b_token_index != real_idx_2:
                raise Exception("tokenization issue: check answer choice tokens!")

    if "vocab_projection_finegrained" in args.function:
        # add hooks for fine-grained projections
        if "vocab_projection_finegrained_attnHeads" in args.function:
            wrapper.add_hooks(type="attn_heads")
            keyname = "in_attn_heads"
            if "olmo" in args.model:
                n_heads = wrapper.model.model.config.n_heads  # 32 in 7B model
                d_model = wrapper.model.model.config.d_model  # 4096 in 7B model
            elif "llama" in args.model:
                n_heads = wrapper.model.model.config.num_attention_heads
                d_model = wrapper.model.model.config.hidden_size
            attn_d = d_model // n_heads
        elif "vocab_projection_finegrained_mlpOutput" in args.function:
            wrapper.add_hooks(type="mlp")
            keyname = "out_mlp"
        elif "vocab_projection_finegrained_attnOutput" in args.function:
            wrapper.add_hooks(type="attention")
            keyname = "out_attn"
        else:
            raise Exception("invalid function type")

    for i, item in tqdm(
        enumerate(knowns[already_completed:], start=already_completed),
        total=len(knowns) - already_completed,
    ):
        start_time = time.time()
        if "trace" in args.function:
            if args.function == "trace_all":
                kinds = ["mlp", "attn", None]
            else:
                kinds = [kind]
            for kind in kinds:
                trace_hidden_flow_and_save(
                    mt,
                    i,
                    prompt=item["prompt"],
                    continuation_1=item["completion_one"],
                    continuation_2=item["completion_two"],
                    records_filename=records_filename,
                    scoring_type=args.scoring_type,
                    initial_prompt=args.base_prompt,
                    prompt_to_sub_in=args.prompt_to_sub_in,
                    kind=kind,
                    include_negatives=args.include_negatives,
                    icl=icl if args.dataset != "mmlu" else icl[item["task"]],
                    icl_ordering=args.icl_labels,
                    override=args.override,
                )
            traced += 1
        elif "calculate_acc" in args.function:
            (
                out,
                formatted_prompt,
                correct_answer_choice,
                incorrect_answer_choice,
                base_probs,
                _,
                top_k_probit_tokens,
                correct_answer_rank,
                incorrect_answer_rank,
                correct_answer_prob,
                incorrect_answer_prob,
                correct_answer_logit,
                incorrect_answer_logit,
            ) = compute_accuracy(
                wrapper,
                prompt=item["prompt"],
                continuation_1=item["completion_one"],
                continuation_2=item["completion_two"],
                scoring_type=args.scoring_type,
                inverse=args.inverse,
                icl=icl if args.dataset != "mmlu" else icl[item["task"]],
                icl_ordering=args.icl_labels,
                cd_label_mapping=args.cd_label_mapping,
            )
            ind_corrs.append(out)

            if i == 0 or i == already_completed:
                if args.override:
                    f = open(acc_file, "w")
                else:
                    f = open(acc_file, "a")
                writer = csv.writer(f)
                if not os.path.isfile(acc_file) or args.override or already_completed == 0:
                    # first row-- also write header
                    writer.writerow(
                        [
                            "index",
                            "prompt",
                            "continuation_1_correct",
                            "continuation_2_incorrect",
                            "continuation_1_score",
                            "continuation_2_score",
                            "score_difference",
                            "correctness",
                            "top_k_preds",
                            "continuation_1_rank",
                            "continuation_2_rank",
                            "continuation_1_prob",
                            "continuation_2_prob",
                            "continuation_1_logit",
                            "continuation_2_logit",
                            "inverse_prompt",
                            "inverse_continuation_1_correct",
                            "inverse_continuation_2_incorrect",
                            "inverse_continuation_1_score",
                            "inverse_continuation_2_score",
                            "inverse_score_difference",
                            "inverse_correctness",
                            "inverse_top_k_preds",
                            "inverse_continuation_1_rank",
                            "inverse_continuation_2_rank",
                            "inverse_continuation_1_prob",
                            "inverse_continuation_2_prob",
                            "inverse_continuation_1_logit",
                            "inverse_continuation_2_logit",
                        ]
                    )

            # also compute instance predictions when answer choice order is swapped
            if args.scoring_type == "enumerated":
                (
                    inv_out,
                    inv_formatted_prompt,
                    inv_correct_answer_choice,
                    inv_incorrect_answer_choice,
                    inv_probs,
                    _,
                    inv_top_k_probit_tokens,
                    inv_correct_answer_rank,
                    inv_incorrect_answer_rank,
                    inv_correct_answer_prob,
                    inv_incorrect_answer_prob,
                    inv_correct_answer_logit,
                    inv_incorrect_answer_logit,
                ) = compute_accuracy(
                    wrapper,
                    prompt=item["prompt"],
                    continuation_1=item["completion_one"],
                    continuation_2=item["completion_two"],
                    scoring_type=args.scoring_type,
                    inverse=(not args.inverse),
                    icl=icl if args.dataset != "mmlu" else icl[item["task"]],
                    icl_ordering=args.icl_labels,
                    cd_label_mapping=args.cd_label_mapping,
                )
                inv_ind_corrs.append(inv_out)

                writer.writerow(
                    [
                        i,
                        formatted_prompt.replace("\n", "\\n"),
                        correct_answer_choice,
                        incorrect_answer_choice,
                        base_probs[1],
                        base_probs[2],
                        base_probs[0],
                        out,
                        top_k_probit_tokens,
                        correct_answer_rank,
                        incorrect_answer_rank,
                        correct_answer_prob,
                        incorrect_answer_prob,
                        correct_answer_logit,
                        incorrect_answer_logit,
                        inv_formatted_prompt.replace("\n", "\\n"),
                        inv_correct_answer_choice,
                        inv_incorrect_answer_choice,
                        inv_probs[1],
                        inv_probs[2],
                        inv_probs[0],
                        inv_out,
                        inv_top_k_probit_tokens,
                        inv_correct_answer_rank,
                        inv_incorrect_answer_rank,
                        inv_correct_answer_prob,
                        inv_incorrect_answer_prob,
                        inv_correct_answer_logit,
                        inv_incorrect_answer_logit,
                    ]
                )
                if out == 1 and inv_out == 1:
                    cons_correct.append(1)
                elif out == 0 and inv_out == 0:
                    cons_incorr.append(1)
                elif out == 1 and inv_out == 0:
                    incons_A.append(1)
                elif out == 0 and inv_out == 1:
                    incons_B.append(1)
                else:
                    raise Exception("invalid predictions")
            else:
                writer.writerow(
                    [
                        i,
                        formatted_prompt.replace("\n", "\\n"),
                        correct_answer_choice,
                        incorrect_answer_choice,
                        base_probs[1],
                        base_probs[2],
                        base_probs[0],
                        out,
                        top_k_probit_tokens,
                        correct_answer_rank,
                        incorrect_answer_rank,
                        correct_answer_prob,
                        incorrect_answer_prob,
                        correct_answer_logit,
                        incorrect_answer_logit,
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                    ]
                )
        elif "vocab_projection" in args.function:
            # run 2 instances (base prompt and prompt to sub in states from) through as a batch
            (
                inp,  # dict("input_ids": torch.tensor [batch size, num_tokens])
                answers_t,  # tuple of formatted prompt's (correct, incorrect) answer indices
                icl_tokenized_length,  # int
                formatted_prompt,  # string
                l1,  # correct answer str
                l2,  # incorrect answer str or list (if multiple)
            ) = encode(
                mt.tokenizer,
                prompt=item["prompt"],
                continuation_1=item["completion_one"],
                continuation_2=item["completion_two"],
                scoring_type=args.scoring_type,
                inverse=args.inverse,
                icl=icl if args.dataset != "mmlu" else icl[item["task"]],
                icl_ordering=args.icl_labels,
                cd_label_mapping=args.cd_label_mapping,
                accuracy_only=True,
            )
            if args.scoring_type == "next_token":
                # get the token indices of the specific color to check for
                a_token_index, b_token_index = answers_t[0], answers_t[1]

            input_ids = inp["input_ids"].to("cuda:1" if torch.cuda.device_count() > 1 else "cuda")
            if "vocab_projection_coarse" in args.function:
                # assumes LayerNorm already applied on last layer
                logits = wrapper.get_layers(input_ids)
                # skip the embedding layer to match outputs of fine-grained case
                # logits.shape = [num_layers, vocab_size, batch_size], e.g., [41, 32001, 2], at the last token position
                logits = logits[1:]
            else:
                # first, reset activations_ dictionary
                wrapper.model.activations_ = {}

                # run inference to activate hooks
                # this populates the wrapper.model.activations_ dictionary
                out = wrapper.model(input_ids=input_ids)
                if "llama2-7b" in args.model or ("olmo" in args.model and "7b" in args.model):
                    dim = 4096
                elif "llama2-13b" in args.model:
                    dim = 5120
                else:
                    raise Exception("must specify hidden dim length for this model")
                for k, v in wrapper.model.activations_.items():
                    if wrapper.model.activations_[k].shape != torch.Size([1, dim]):
                        print(k)

                hidden_states = []
                # reformat/reshape hidden states for vocab projection function
                for j in range(wrapper.num_layers):
                    if (
                        "vocab_projection_finegrained_attnHeads" in args.function
                        and i >= completed_per_layer[j]
                    ):
                        # get each individual head, and take product with attention output matrix
                        # this implements Equation 3 in Appendix A.3 of the paper
                        layerwise_filename = records_filename.replace("layer", f"layer{j}")
                        if "llama2" in args.model:
                            attn_output_weights = wrapper.model.model.layers[
                                j
                            ].self_attn.o_proj.weight
                        elif "olmo" in args.model:
                            attn_output_weights = wrapper.model.model.transformer.blocks[
                                j
                            ].attn_out.weight
                        else:
                            raise Exception("model not supported")
                        if attn_output_weights.shape != torch.Size([d_model, d_model]):
                            print(attn_output_weights.shape)
                            breakpoint()

                        int_attn_output = wrapper.model.activations_[f"{keyname}_{j}"]
                        if int_attn_output.shape != torch.Size([1, d_model]):
                            print(int_attn_output.shape)
                            breakpoint()

                        per_layer_hidden_states = []
                        for head_idx in range(n_heads):
                            # get index of relevant intermediate states and outputs
                            start_idx = head_idx * attn_d
                            end_idx = (head_idx + 1) * attn_d
                            if i == 0 and j == 0:
                                print(f"Head {head_idx}: ({start_idx}, {end_idx})")
                            # get the attention output for this head
                            int_attn_output_head = int_attn_output[:, start_idx:end_idx]
                            if int_attn_output_head.shape != torch.Size([1, attn_d]):
                                print(int_attn_output_head.shape)
                                breakpoint()

                            # Pytorch documentation: nn.Linear() "applies a linear transformation to the incoming data as y = xA^T + b"
                            # dimensions of weight: (out_features, in_features)
                            # therefore, for any head, we want to take the subset of *columns* pre-transpose, or *rows* post-transpose
                            head_specific_weights = attn_output_weights[:, start_idx:end_idx]
                            if head_specific_weights.shape != torch.Size([d_model, attn_d]):
                                print(head_specific_weights.shape)
                                breakpoint()

                            # get output of specific attention head
                            head_specific_output = torch.matmul(
                                int_attn_output_head, head_specific_weights.T
                            )
                            # should now be back to original dimensions [1, 4096]
                            if head_specific_output.shape != torch.Size([1, d_model]):
                                print(head_specific_output.shape)
                                breakpoint()

                            # reshape the output to follow the convention of the layer_decode method; then pass in to do the final LN + vocab projection
                            # layer_logits.shape = [|V|, batch_size] = [32001, 1] for Llama2-7b, [50304, 1] for Olmo-7B
                            # does NOT assume LayerNorm has already been applied
                            layer_logits = wrapper.layer_decode(
                                [head_specific_output.unsqueeze(0)]
                            )[0]

                            # for each attn head (32 per layer), append to sub-list of hidden_states
                            per_layer_hidden_states.append(layer_logits)

                        # convert each instance's projections to a Tensor of shape [num_heads, vocab_size, batch_size]
                        logits = torch.stack(per_layer_hidden_states)

                        # write out stats for each attention head in each layer
                        top_k_per_item_probs = wrapper.topk_per_layer(logits, k=args.k, log=False)
                        top_k_per_item_logs = wrapper.topk_per_layer(logits, k=args.k, log=True)
                        rr_a = wrapper.rr_per_layer(logits, a_token_index)
                        a_probs = wrapper.prob_of_answer_per_layer(logits, a_token_index)
                        a_logs = wrapper.log_of_answer_per_layer(logits, a_token_index)
                        if args.scoring_type == "enumerated":
                            rr_b = wrapper.rr_per_layer(logits, b_token_index)
                            b_probs = wrapper.prob_of_answer_per_layer(logits, b_token_index)
                            b_logs = wrapper.log_of_answer_per_layer(logits, b_token_index)
                            if args.cd_label_mapping:
                                real_a_probs = wrapper.prob_of_answer_per_layer(
                                    logits, real_a_token_index
                                )
                                real_b_probs = wrapper.prob_of_answer_per_layer(
                                    logits, real_b_token_index
                                )
                                real_a_logs = wrapper.log_of_answer_per_layer(
                                    logits, real_a_token_index
                                )
                                real_b_logs = wrapper.log_of_answer_per_layer(
                                    logits, real_b_token_index
                                )
                        elif args.scoring_type == "next_token":
                            # go through each element in the list of b_token_index (incorrect answer strings), and sum their probabilities
                            rr_b, b_probs, b_logs = (
                                [defaultdict(float)],
                                [0] * wrapper.num_layers,
                                [0] * wrapper.num_layers,
                            )
                            for el in b_token_index:
                                tmp_rr_b = wrapper.rr_per_layer(logits, el)[0]
                                tmp_b_probs = wrapper.prob_of_answer_per_layer(logits, el)
                                tmp_b_logs = wrapper.log_of_answer_per_layer(logits, el)
                                for layer in range(wrapper.num_layers):
                                    # sum of probabilities assigned to other colors
                                    rr_b[0][layer] += tmp_rr_b[layer]
                                    b_probs[layer] += tmp_b_probs[layer]
                                    b_logs[layer] += tmp_b_logs[layer]

                        if logits.shape[2] != 1:
                            print("code not designed to work for batch sizes >1")
                            breakpoint()

                        # create individual instance for writing out to jsonl file
                        inst = {
                            "vocab_projections_probs": top_k_per_item_probs[0],
                            "vocab_projections_logs": top_k_per_item_logs[0],
                            "a_ranks": rr_a[0],
                            "b_ranks": rr_b[0],
                            "a_probs": a_probs,
                            "b_probs": b_probs,
                            "real_a_probs": real_a_probs if args.cd_label_mapping else None,
                            "real_b_probs": real_b_probs if args.cd_label_mapping else None,
                            "real_a_logs": real_a_logs if args.cd_label_mapping else None,
                            "real_b_logs": real_b_logs if args.cd_label_mapping else None,
                            "a_logs": a_logs,
                            "b_logs": b_logs,
                            "prompt": formatted_prompt,
                            "idx": i,
                            "input_ids": inp["input_ids"][0],
                            "input_tokens": decode_tokens(wrapper.tokenizer, inp["input_ids"][0]),
                            "continuations": [
                                list(item["completion_one"]),
                                list(item["completion_two"]),
                            ],
                            "answer_tokens": (l1, l2),
                            "answer_ids": answers_t,
                            "icl_length": icl_tokenized_length,
                        }
                        # write to file
                        write_out(layerwise_filename, inst, override=args.override)

                    else:
                        # reshape from [batch_size, hidden_state_dim] to [batch_size, 1, hidden_state_dim]
                        hidden_states.append(
                            wrapper.model.activations_[f"{keyname}_{j}"].unsqueeze(1)
                        )

                if "vocab_projection_finegrained_attnHeads" not in args.function:
                    # do vocabulary projections
                    # hidden states = tuple of, e.g., len(40) for 40 model layers
                    # hidden_states[0].shape = e.g., [2, 221, 5120] for [batch_size, input_sequence_length, hidden_state_dim]
                    # does NOT assume LayerNorm has already been applied
                    logits = wrapper.layer_decode(tuple(hidden_states))
                    # logits = list of len(num_layers) of tensors of shape (vocab_size, batch_size)

                    # convert back to a tensor of shape (num_layers-1, vocab_size, batch_size); skip the embedding layer
                    logits = torch.stack(logits)

            # have already saved on a per-layer basis for attention head projections
            if "vocab_projection_finegrained_attnHeads" not in args.function:
                # here, result is a list of len(batch_size) containing dictionaries where keys are layer integers and values are lists of tuples of top-k tokens (token, probability)
                top_k_per_item_probs = wrapper.topk_per_layer(logits, k=args.k, log=False)
                top_k_per_item_logs = wrapper.topk_per_layer(logits, k=args.k, log=True)
                rr_a = wrapper.rr_per_layer(logits, a_token_index)
                a_probs = wrapper.prob_of_answer_per_layer(logits, a_token_index)
                a_logs = wrapper.log_of_answer_per_layer(logits, a_token_index)
                if args.scoring_type == "enumerated":
                    rr_b = wrapper.rr_per_layer(logits, b_token_index)
                    b_probs = wrapper.prob_of_answer_per_layer(logits, b_token_index)
                    b_logs = wrapper.log_of_answer_per_layer(logits, b_token_index)
                    if args.cd_label_mapping:
                        real_a_probs = wrapper.prob_of_answer_per_layer(logits, real_a_token_index)
                        real_b_probs = wrapper.prob_of_answer_per_layer(logits, real_b_token_index)
                        real_a_logs = wrapper.log_of_answer_per_layer(logits, real_a_token_index)
                        real_b_logs = wrapper.log_of_answer_per_layer(logits, real_b_token_index)
                elif args.scoring_type == "next_token":
                    # go through each element in the list of b_token_index, and sum their probabilities
                    rr_b, b_probs, b_logs = (
                        [defaultdict(float)],
                        [0] * wrapper.num_layers,
                        [0] * wrapper.num_layers,
                    )
                    for el in b_token_index:
                        tmp_rr_b = wrapper.rr_per_layer(logits, el)[0]
                        tmp_b_probs = wrapper.prob_of_answer_per_layer(logits, el)
                        tmp_b_logs = wrapper.log_of_answer_per_layer(logits, el)
                        for layer in range(wrapper.num_layers):
                            # sum of probabilities assigned to other colors
                            rr_b[0][layer] += tmp_rr_b[layer]
                            b_probs[layer] += tmp_b_probs[layer]
                            b_logs[layer] += tmp_b_logs[layer]

                if logits.shape[2] != 1:
                    print("code not designed to work for batch sizes >1")
                    breakpoint()

                # create individual instance
                inst = {
                    "vocab_projection_probs": top_k_per_item_probs[0],
                    "vocab_projection_logs": top_k_per_item_logs[0],
                    "a_ranks": rr_a[0],
                    "b_ranks": rr_b[0],
                    "a_probs": a_probs,
                    "b_probs": b_probs,
                    "real_a_probs": real_a_probs if args.cd_label_mapping else None,
                    "real_b_probs": real_b_probs if args.cd_label_mapping else None,
                    "real_a_logs": real_a_logs if args.cd_label_mapping else None,
                    "real_b_logs": real_b_logs if args.cd_label_mapping else None,
                    "a_logs": a_logs,
                    "b_logs": b_logs,
                    "prompt": formatted_prompt,
                    "idx": i,
                    "input_ids": inp["input_ids"][0],
                    "input_tokens": decode_tokens(wrapper.tokenizer, inp["input_ids"][0]),
                    "continuations": [
                        list(item["completion_one"]),
                        list(item["completion_two"]),
                    ],
                    "answer_tokens": (l1, l2),
                    "answer_ids": answers_t,
                    "icl_length": icl_tokenized_length,
                }
                # write to file
                write_out(records_filename, inst, override=args.override)

        else:
            raise Exception("invalid function type")

        lt = round((time.time() - start_time) / 60.0, 2)
        times.append(lt)

    if "calculate_acc" in args.function:
        print("Number individual datapoints correct: ", sum(ind_corrs))
        print(
            "Acc of individual datapoints correct: ",
            sum(ind_corrs) / float(len(ind_corrs)),
        )
        if args.scoring_type == "enumerated":
            # also return other stats
            print("Number INVERTED individual datapoints correct: ", sum(inv_ind_corrs))
            print(
                "INVERTED Acc of individual datapoints correct: ",
                sum(inv_ind_corrs) / float(len(inv_ind_corrs)),
            )
            print("Number consistent correct: ", sum(cons_correct))
            print("Number consistent incorrect: ", sum(cons_incorr))
            print("Number inconsistent A correct: ", sum(incons_A))
            print("Number inconsistent B correct: ", sum(incons_B))
            print("Percent consistent correct: ", sum(cons_correct) / float(len(ind_corrs)))
            print("Percent consistent incorrect: ", sum(cons_incorr) / float(len(ind_corrs)))
            print("Percent inconsistent A correct: ", sum(incons_A) / float(len(ind_corrs)))
            print("Percent inconsistent B correct: ", sum(incons_B) / float(len(ind_corrs)))
            print(
                f"{sum(ind_corrs) / float(len(ind_corrs))}/{sum(inv_ind_corrs) / float(len(inv_ind_corrs))}"
            )
            print(
                f"{sum(cons_correct) / float(len(ind_corrs))}/{sum(cons_incorr) / float(len(ind_corrs))}/{sum(incons_A) / float(len(ind_corrs))}/{sum(incons_B) / float(len(ind_corrs))}"
            )

    if "trace" in args.function:
        print(f"Traced {traced} instances")

    print("AVERAGE LOOP TIME: %.4f minutes" % (round(sum(times) / len(times), 2)))
    print("TOTAL SCRIPT TIME: %.4f hours" % (round((time.time() - og_start_time) / 60.0 / 60.0, 2)))
