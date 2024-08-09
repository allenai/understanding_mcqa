import os
import re

# matplotlib produces a lot of warnings-- silence them
import warnings
from collections import defaultdict

import jsonlines
import torch
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer

from model.modeling_modified_olmo import ModifiedOLMoForCausalLM
from util import nethook

warnings.filterwarnings("ignore")

"""
This file is modified from the MEMIT causal tracing file provided by Kevin Meng (https://github.com/kmeng01/memit/blob/main/experiments/causal_trace.py),
namely, the "trace_with_patch" and "make_inputs" functions and the "ModelAndTokenizer" class.
"""


def trace_with_patch(
    mt,  # The model and tokenizer
    inp,  # A dict of inputs with key "input_ids" and value a torch.tensor of shape [batch, num_tokens]
    answers_t,  # Answer probabilities to collect- depending on scoring type, this is either 1 or 2 encoded token strings
    indices_to_replace,
    kind,
):
    """
    Runs all causal traces specified for a single instance.  Given a model
    and a batch input where the batch size is two, runs batch instance [0]
    in inference, restoring a set of hidden states to the values from another
    run [1] representing the counterfactual instance (opposite correct answer).

    The convention used by this function is that the zeroth element of the
    batch is the base instance, and the first element in the new instance.

    Then when running, a specified set of hidden states will be changed
    by changing their values to the vector that they have in the
    first run. This set of hidden states is listed in states_to_patch, by listing
    [(token_index, layername), ...] pairs. To trace the effect of just a single state, this can be just a single
    token/layer pair.  To trace the effect of restoring a set of states,
    any number of token indices and layers can be listed.
    """
    if indices_to_replace is None:
        # just replace the last index
        indices_to_replace = [inp["input_ids"].shape[1] - 1]

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep_seqScoring(x, layer):
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, insert the counterfactual's hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec[layer]:
            # replace from other run (position 1) at token position t
            h[0, t] = h[1, t]
        return x

    # with the patching rule defined, run the patched model in inference, transferring all the states to be patched at each layer.
    probs_table_corr, probs_table_incorr, top_k_tokens = [], [], []
    other_token_probs = defaultdict(list)
    for layer in range(mt.num_layers):
        lname = layername(mt, layer, kind)

        # specify what to patch: a list of (token index, layername) triples to restore
        states_to_patch = [(token_pos, lname) for token_pos in indices_to_replace]

        patch_spec = defaultdict(list)
        for t, l in states_to_patch:
            patch_spec[l].append(t)

        # run activation patching
        with torch.inference_mode(), nethook.TraceDict(
            mt.model,
            list(patch_spec.keys()),
            edit_output=patch_rep_seqScoring,
        ):
            p, l, other_p, top_k = score_target(
                mt,
                inp,
                answers_t,
                counterfactual=False,
            )

        # p is a tuple of (probs, corr_prob, incorr_prob)
        # probs is something like 0.0948 (an individual score = corr_prob - incorr_prob)
        probs_table_corr.append(p[1])
        probs_table_incorr.append(p[2])
        if len(other_p) != 2:
            breakpoint()
        other_token_probs[answers_t[2]].append(other_p[0])
        other_token_probs[answers_t[3]].append(other_p[1])
        top_k_tokens.append(top_k)

    # these are all lists of L floats (for L layers in model)
    return probs_table_corr, probs_table_incorr, other_token_probs, top_k_tokens


def encode(
    tokenizer,
    prompt,
    continuation_1,
    continuation_2,
    scoring_type,
    accuracy_only=False,
    inverse=False,
    icl=None,
    icl_ordering=None,
    cd_label_mapping=False,
    prompt_to_sub_in=None,
):

    if scoring_type == "enumerated":
        if cd_label_mapping:
            label1 = "C"
            label2 = "D"
        else:
            label1 = "A"
            label2 = "B"

        if prompt_to_sub_in is not None:
            # also get labels for second instance
            if "cd_label_mapping" in prompt_to_sub_in:
                label3 = "C"
                label4 = "D"
            else:
                label3 = "A"
                label4 = "B"

        # parse in-context label ordering
        if icl is not None and len(icl) > 0:
            assert isinstance(icl_ordering, str)
            ordering = []
            for el in icl_ordering:
                if el not in {"A", "B"}:
                    raise Exception("invalid ordering")
                ordering.append(0 if el == "A" else 1)
            if len(ordering) != len(icl):
                print("specified ICL labeling results in fewer in-context examples.")
        else:
            ordering = []

        # construct string of in-context examples in appropriate format
        icl_string = f"For each of the following phrases, select the best completion ({label1} or {label2}).\n\n"
        for z, it in enumerate(icl[: len(ordering)]):
            a1, a2 = it["completion_one"][0], it["completion_two"][0]
            if ordering[z] == 0:
                # first answer choice is correct; append correct answer choice
                icl_string += (
                    f"Phrase: {it['prompt']} {it['completion_one'][1]}".strip()
                    + f"\nChoices:\n{label1}: {a1}\n{label2}: {a2}\nThe correct answer is: {label1}\n\n"
                )
            else:
                # second answer choice is correct; append correct answer choice
                icl_string += (
                    f"Phrase: {it['prompt']} {it['completion_one'][1]}".strip()
                    + f"\nChoices:\n{label1}: {a2}\n{label2}: {a1}\nThe correct answer is: {label2}\n\n"
                )

        # encode prompts
        a1, a2 = continuation_1[0], continuation_2[0]
        if inverse:
            formatted_prompt = (
                icl_string
                + f"Phrase: {prompt} {continuation_1[1]}".strip()
                + f"\nChoices:\n{label1}: {a2}\n{label2}: {a1}\nThe correct answer is:"
            )
        else:
            formatted_prompt = (
                icl_string
                + f"Phrase: {prompt} {continuation_1[1]}".strip()
                + f"\nChoices:\n{label1}: {a1}\n{label2}: {a2}\nThe correct answer is:"
            )
        if prompt_to_sub_in is not None:
            if "inverse" in prompt_to_sub_in:
                # contrast example should be inverted
                contrast_prompt = (
                    icl_string
                    + f"Phrase: {prompt} {continuation_2[1]}".strip()
                    + f"\nChoices:\n{label3}: {a2}\n{label4}: {a1}\nThe correct answer is:"
                )
            else:
                contrast_prompt = (
                    icl_string
                    + f"Phrase: {prompt} {continuation_2[1]}".strip()
                    + f"\nChoices:\n{label3}: {a1}\n{label4}: {a2}\nThe correct answer is:"
                )

    elif scoring_type == "next_token":
        assert accuracy_only
        label1, label2 = continuation_1[0], continuation_2[0]  # a list

        # construct string of in-context examples
        icl_string = ""
        for it in icl[: len(icl_ordering)]:
            icl_string += f"{it['prompt']} {it['completion_one'][0]}\n"

        # encode prompts
        formatted_prompt = icl_string + prompt

    if (
        continuation_2[1] != ""
        or continuation_1[1] != ""
        or (not accuracy_only and len(formatted_prompt) != len(contrast_prompt))
    ):
        raise Exception("this will cause difficulty for causal tracing")

    if accuracy_only:
        full_input_strings = [formatted_prompt]
    else:
        # also include contrast prompt
        full_input_strings = [formatted_prompt, contrast_prompt]

    # creates 2-dimensional object on flattened input list: [# sequences * 2, (padded) # tokens]
    # padding isn't needed for single_token scoring, since prompts will be the same length by design (also good because LLaMA doesn't have a pad token)
    inp = make_inputs(tokenizer, full_input_strings)
    icl_tokenized_length = len(make_inputs(tokenizer, [icl_string])["input_ids"][0])

    # get exact token indices of the answer choice
    # to properly encode, let's append to the full input during encoding, and then isolate the answer token ids
    if inverse:
        full_input_strings_with_answers = [
            formatted_prompt + " " + label2,
            formatted_prompt + " " + label1,
        ]
    else:
        if scoring_type == "next_token":
            # append each alternative color label
            full_input_strings_with_answers = [formatted_prompt + " " + label1] + [
                formatted_prompt + " " + label2[el] for el in range(len(label2))
            ]
        else:
            full_input_strings_with_answers = [
                formatted_prompt + " " + label1,
                formatted_prompt + " " + label2,
            ]
    if prompt_to_sub_in is not None:
        if "inverse" in prompt_to_sub_in:
            full_input_strings_with_answers += [
                contrast_prompt + " " + label4,
                contrast_prompt + " " + label3,
            ]
        else:
            full_input_strings_with_answers += [
                contrast_prompt + " " + label3,
                contrast_prompt + " " + label4,
            ]

    full_input_encodings_with_answers = make_inputs(
        tokenizer,
        full_input_strings_with_answers,
        truncate=True if scoring_type == "next_token" else False,
    )
    if full_input_encodings_with_answers["input_ids"].shape[1] != inp["input_ids"].shape[1] + 1:
        raise Exception(
            'Answer tokens are >1 token-- check that this is correct; it shouldn\'t be for "A" and "B".'
        )
    # get (first) answer choice tokens only
    answer_encodings = [
        el[0]
        for el in full_input_encodings_with_answers["input_ids"][
            :, inp["input_ids"].shape[1] :
        ].tolist()
    ]
    # ensure they are in the correct set
    for el in answer_encodings:
        if scoring_type == "enumerated" and el not in {315, 360, 319, 350, 330, 399, 329, 378}:
            breakpoint()
        elif scoring_type == "next_token" and el not in [
            13328,
            7254,
            18345,
            24841,
            7933,
            3708,
            4628,
            17354,
            2654,
            4796,
            282,
        ] + [
            8862,
            4759,
            2806,
            14863,
            19445,
            3168,
            13735,
            4797,
            8516,
            2502,
            14370,
        ]:  # Llama + Olmo tokenizations
            breakpoint()
    if not accuracy_only:
        # also ensure that the answer choices are not identical if doing causal tracing
        if answer_encodings[0] == answer_encodings[2] or answer_encodings[1] == answer_encodings[3]:
            raise Exception(
                "PROBLEMATIC CAUSAL TRACING PAIR -- SAME LABELS, AND SAME CORRECT LABEL"
            )

    if accuracy_only:
        return (
            inp,
            (
                answer_encodings[0],
                answer_encodings[1:] if scoring_type == "next_token" else answer_encodings[1],
            ),  # formatted_prompt's correct, incorrect answer indices
            icl_tokenized_length,
            # for writing out to .csv file:
            formatted_prompt,
            label2 if inverse else label1,
            label1 if inverse else label2,
        )
    else:
        return (
            inp,
            answer_encodings,  # formatted_prompt's (correct, incorrect) answer indices + contrast's (correct, incorrect) answer indices, if they exist
            None,  # tuple of start and end indices of first answer choice
            None,  # tuple of start and end indices of second answer choice
            None,
            icl_tokenized_length,
            formatted_prompt,
            contrast_prompt if prompt_to_sub_in is not None else None,
        )


def trace_hidden_flow(
    mt,
    prompt,
    continuation_1,
    continuation_2,
    scoring_type,
    initial_prompt,
    prompt_to_sub_in,
    kind=None,
    include_negatives=False,
    icl=None,
    icl_ordering=None,
):
    """
    Runs causal tracing over every layer at the last token position in the network
    and returns a dictionary numerically summarizing the results.
    """

    (
        inp,  # dict("input_ids": torch.tensor [batch size, num_tokens])
        answers_t,  # list of formatted prompt's (correct, incorrect) answer indices, and optionally contrast's (correct, incorrect) answer indices
        _,  # tuple(int, int) of start and end indices of first answer choice
        _,  # tuple(int, int) of start and end indices of second answer choice
        _,  # int
        icl_tokenized_length,  # int
        _,  # formatted prompt (str)
        _,  # contrast prompt (str)
    ) = encode(
        mt.tokenizer,
        prompt,
        continuation_1,
        continuation_2,
        scoring_type,
        accuracy_only=False,
        icl=icl,
        icl_ordering=icl_ordering,
        inverse=True if "inverse" in initial_prompt else False,
        cd_label_mapping=True if "cd_label_mapping" in initial_prompt else False,
        prompt_to_sub_in=prompt_to_sub_in,
    )

    with torch.inference_mode():
        # Initial run: runs model inference, performs softmax on logits, and returns the difference between the correct and incorrect answer probabilities
        base_inst, counterfact_instance = score_target(
            mt,
            inp,
            answers_t,
            counterfactual=True,
        )
        base_prob_diff = base_inst[0][0]
        counterfact_prob_diff = counterfact_instance[0][0]

    # correct_prediction if answer is predicted correctly for BOTH variants
    if base_prob_diff <= 0 or counterfact_prob_diff <= 0:
        if base_prob_diff <= 0 and counterfact_prob_diff > 0:
            prediction_type = "contrast_correct"
        elif base_prob_diff > 0 and counterfact_prob_diff <= 0:
            prediction_type = "base_correct"
        else:
            prediction_type = "both_incorrect"
        if not include_negatives:
            # do not trace this instance
            return dict(correct_prediction=False)
    else:
        # proceed with tracing
        prediction_type = "both_correct"

    # Run with patching: this gives some list of ~30 values corr. to layers & token positions
    prob_corr, prob_incorr, other_token_probs, top_k_tokens = trace_with_patch(
        mt=mt,
        inp=inp,
        answers_t=answers_t,
        indices_to_replace=None,  # only replace the last token position
        kind=kind,
    )
    return dict(
        probits_correct=prob_corr,
        probits_incorrect=prob_incorr,
        other_token_probs=other_token_probs,
        top_k_tokens=top_k_tokens,
        base_probs_logs=base_inst,
        contrast_probs_logs=counterfact_instance,
        input_ids_base_inst=inp["input_ids"][0].tolist(),
        input_ids_counterfact_inst=inp["input_ids"][1].tolist(),
        input_tokens_base_inst=mt.tokenizer.decode(inp["input_ids"][0]),
        input_tokens_counterfact_inst=mt.tokenizer.decode(inp["input_ids"][1]),
        base_answer_tokens=(mt.tokenizer.decode(answers_t[0]), mt.tokenizer.decode(answers_t[1])),
        prediction_type=prediction_type,
        kind=kind,
        icl_length=icl_tokenized_length,
        base_corr_incorr_answer_ids=answers_t,
    )


class ModelAndTokenizer:
    """
    An object to hold on to (or automatically download and hold)
    a language model and tokenizer.  Counts the number
    of layers.
    """

    def __init__(
        self,
        model_name,
        model=None,
        tokenizer=None,
        no_model_load=False,
        llama_path=None,
    ):
        if tokenizer is None:
            assert model_name is not None
            if "olmo-" in model_name:
                # load base or FT 7B model
                assert "7b" in model_name
                if "v1.7" in model_name:
                    # load v1.7 tokenizer
                    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B-0424")
                elif "sft" in model_name:
                    # load v1 SFT tokenizer
                    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B-SFT")
                else:
                    # load v1 tokenizer
                    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B")
            elif "llama2-" in model_name:
                # load either base or chat model of specified size
                # for some reason, LLama models don't work with Auto modules
                size = "-".join(model_name.split("-")[1:]).replace("b", "B")
                assert size in {"7B", "13B", "7B-chat", "13B-chat"}
                tokenizer = LlamaTokenizer.from_pretrained(os.path.join(llama_path, size))
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
        if no_model_load:
            model = None
        elif model is None:
            assert model_name is not None
            if "olmo-" in model_name:
                # load base or FT 7B model
                assert "7b" in model_name
                if "v1.7" in model_name:
                    if "350B" in model_name:
                        # load intermediate checkpoint
                        model = ModifiedOLMoForCausalLM.from_pretrained(
                            "allenai/OLMo-7B-0424", revision="step83500-tokens350B"
                        )
                    elif "1T" in model_name:
                        # load intermediate checkpoint
                        model = ModifiedOLMoForCausalLM.from_pretrained(
                            "allenai/OLMo-7B-0424", revision="step239000-tokens1002B"
                        )
                    else:
                        # load final checkpoint
                        model = ModifiedOLMoForCausalLM.from_pretrained("allenai/OLMo-7B-0424")
                elif "sft" in model_name:
                    model = ModifiedOLMoForCausalLM.from_pretrained("allenai/OLMo-7B-SFT")
                else:
                    model = ModifiedOLMoForCausalLM.from_pretrained("allenai/OLMo-7B")
                if torch.cuda.is_available():
                    # manually place on device
                    # TODO: may not work with multi-GPU support
                    # Olmo doesn't support auto device mapping due to untied embeddings
                    try:
                        model = model.to("cuda:1" if torch.cuda.device_count() > 1 else "cuda")
                    except:
                        raise Exception("model not effectively going on device")
            elif "llama2-" in model_name:
                # load either base or chat model of specified size
                size = "-".join(model_name.split("-")[1:]).replace("b", "B")
                assert size in {"7B", "13B", "7B-chat", "13B-chat"}
                model = LlamaForCausalLM.from_pretrained(
                    os.path.join(llama_path, size), device_map="auto"
                )
            model.resize_token_embeddings(len(tokenizer))
            nethook.set_requires_grad(False, model)
            model.eval()
        if not no_model_load:
            self.model = model
            self.layer_names = [
                n
                for n, m in model.named_modules()
                if (
                    re.match(
                        r"^(transformer|gpt_neox|model)\.(h|layers|transformer\.blocks)\.\d+$", n
                    )
                )
            ]
            self.num_layers = len(self.layer_names)
        else:
            if "13b" in model_name:
                self.num_layers = 40
            elif "7b" in model_name:
                self.num_layers = 32
            else:
                raise Exception("unknown model size")
        self.tokenizer = tokenizer
        # sanity check
        if not no_model_load:
            if ("13b" in model_name and self.num_layers != 40) or (
                "7b" in model_name and self.num_layers != 32
            ):
                breakpoint()

    def __repr__(self):
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )


def layername(mt, num, kind):
    model = mt.model
    if isinstance(model, ModifiedOLMoForCausalLM):
        if kind == "embed":
            layername = "model.transformer.wte"
        else:
            if kind == "attn":
                # this is our custom identity layer to the get the output of the attention block; equivalent to 'attn_out'
                kind = "pseudo_layer_2"
            elif kind == "mlp":
                kind = "ff_out"
            elif kind == "attn_heads":
                breakpoint()
            layername = f'model.transformer.blocks.{num}{"" if kind is None else "." + kind}'
    elif hasattr(model, "transformer"):
        if kind == "embed":
            layername = "transformer.wte"
        else:
            layername = f'transformer.h.{num}{"" if kind is None else "." + kind}'
    elif hasattr(model, "model"):
        if kind == "embed":
            layername = "model.embed_tokens"
        else:
            if kind == "attn":
                kind = "self_attn"
            layername = f'model.layers.{num}{"" if kind is None else "." + kind}'
    else:
        assert False, "unknown transformer structure"

    if layername not in [n for n, _ in mt.model.named_modules()]:
        raise Exception("invalid layername: ", layername)

    return layername


def trace_hidden_flow_and_save(
    mt,
    idx,
    prompt,
    continuation_1,
    continuation_2,
    records_filename,
    scoring_type,
    initial_prompt,
    prompt_to_sub_in,
    kind=None,
    include_negatives=False,
    return_stats=False,
    icl=None,
    icl_ordering=None,
    override=False,
):
    result = trace_hidden_flow(
        mt,
        prompt,
        continuation_1,
        continuation_2,
        scoring_type,
        initial_prompt,
        prompt_to_sub_in,
        kind=kind,
        include_negatives=include_negatives,
        icl=icl,
        icl_ordering=icl_ordering,
    )

    # add other relevant values
    result["idx"] = idx
    result["prompt"] = prompt
    result["continuations"] = [continuation_1, continuation_2]
    result["stats"] = {
        "kind": kind,
        "include_negatives": include_negatives,
    }

    write_out(records_filename, result, kind=kind, override=override)

    if return_stats:
        return result


def write_out(records_filename, result, kind=None, override=False):
    if kind is not None:
        records_filename = records_filename.replace("_all", f"_{kind}")

    # convert all torch tensors to lists for jsonl dump
    for el in [k for k, v in result.items() if isinstance(v, torch.Tensor)]:
        result[el] = result[el].tolist()

    # write to jsonlines file
    with jsonlines.open(records_filename, mode="w" if override else "a") as writer:
        writer.write(result)


def compute_accuracy(
    mt,
    prompt,
    continuation_1,
    continuation_2,
    scoring_type,
    inverse=False,
    icl=None,
    icl_ordering=None,
    cd_label_mapping=False,
):

    (
        inp,
        answers_t,
        _,
        formatted_prompt,
        correct_answer_choice,
        incorrect_answer_choice,
    ) = encode(
        mt.tokenizer,
        prompt,
        continuation_1,
        continuation_2,
        scoring_type,
        accuracy_only=True,
        inverse=inverse,
        icl=icl,
        icl_ordering=icl_ordering,
        cd_label_mapping=cd_label_mapping,
    )

    with torch.inference_mode():
        (
            base_probs,
            base_logs,
            top_k_probit_tokens,
            correct_answer_rank,
            incorrect_answer_rank,
            correct_answer_prob,
            incorrect_answer_prob,
            correct_answer_logit,
            incorrect_answer_logit,
        ) = score_target(
            mt,
            inp,
            answers_t,
            counterfactual=False,
            return_top_k=True,
        )

    # determine if predicted correctly based on scoring rule
    if scoring_type == "enumerated":
        if base_probs[0] <= 0:
            sc = 0
        else:
            sc = 1
    elif scoring_type == "next_token":
        # check whether greedy (top token) is correct
        if top_k_probit_tokens[0][0] == answers_t[0]:
            sc = 1
        else:
            sc = 0

    return (
        sc,
        formatted_prompt,
        correct_answer_choice,
        incorrect_answer_choice,
        base_probs,
        base_logs,
        top_k_probit_tokens,
        correct_answer_rank,
        incorrect_answer_rank,
        correct_answer_prob,
        incorrect_answer_prob,
        correct_answer_logit,
        incorrect_answer_logit,
    )


# Utilities for dealing with tokens
def make_inputs(
    tokenizer,
    prompts,
    device="cuda",
    add_special_tokens=True,
    truncate=False,
):
    token_lists = [tokenizer.encode(p, add_special_tokens=add_special_tokens) for p in prompts]
    input_ids = token_lists

    try:
        r1 = torch.tensor(input_ids)
    except:
        if truncate:
            # chop color tokens off to be single-token
            min_len = min([len(t) for t in input_ids])
            truncated_input_ids = [t[:min_len] for t in input_ids]
            try:
                r1 = torch.tensor(truncated_input_ids)
            except:
                breakpoint()
        else:
            raise Exception("input_ids are not the same length; must specify padding")

    return dict(input_ids=r1.to(device))


def decode_tokens(tokenizer, token_array):
    """
    This function comes from https://github.com/kmeng01/memit/blob/main/experiments/causal_trace.py
    """
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


def score_target(
    mt,
    full_input,
    answers_t,
    return_top_k=False,
    counterfactual=False,
):

    with torch.inference_mode():
        outputs = mt.model(
            input_ids=full_input["input_ids"],
            return_dict=True,
        )

    correct_answer_idx = answers_t[0]
    incorrect_answer_idx = answers_t[1]
    if len(answers_t) > 2:
        correct_contrast_idx = answers_t[2]
        incorrect_contrast_idx = answers_t[3]
    if isinstance(answers_t[1], list):
        incorrect_answer_idx = incorrect_answer_idx[0]
    last_token_logits = outputs.logits[:, -1, :]
    last_token_probs = torch.nn.Softmax(dim=1)(last_token_logits)

    if last_token_logits.shape[0] > 2:
        print("this method curr only returns predictions for 2 instances")

    if return_top_k:
        if counterfactual or last_token_logits.shape[0] > 1:
            raise Exception("this code only works for single instances currently")

        # there is only one batch element; take this batch element's values directly
        logits_that_predict_ans_choice = last_token_logits[0]
        probits_that_predict_ans_choice = last_token_probs[0]

        top_k_probits = torch.topk(probits_that_predict_ans_choice, 10)
        top_k_logits = torch.topk(logits_that_predict_ans_choice, 10)
        if not torch.all(top_k_probits.indices == top_k_logits.indices).item():
            print("CHECK THIS INSTANCE:")
            print(top_k_probits)
            print(top_k_logits)
        top_k_probit_tokens = [
            (a.item(), mt.tokenizer.decode(a), b.item(), c.item())
            for a, b, c in zip(top_k_probits.indices, top_k_probits.values, top_k_logits.values)
        ]
        # also get the rank of the answer choice tokens
        # will also be in order (correct, incorrect)
        sorted_token_logits = logits_that_predict_ans_choice.argsort(descending=True)
        correct_answer_rank = torch.where(sorted_token_logits == correct_answer_idx)[0].item()
        incorrect_answer_rank = torch.where(sorted_token_logits == incorrect_answer_idx)[0].item()
        correct_answer_prob = probits_that_predict_ans_choice[correct_answer_idx].item()
        incorrect_answer_prob = probits_that_predict_ans_choice[incorrect_answer_idx].item()
        correct_answer_logit = logits_that_predict_ans_choice[correct_answer_idx].item()
        incorrect_answer_logit = logits_that_predict_ans_choice[incorrect_answer_idx].item()

    # get the probits of the answer choices in batch at the last token position and
    # subtract correct from incorrect to get a score
    # note that (correct_label, incorrect_label) is always the order of the predictions, not ("A", "B")
    base_corr_prob = last_token_probs[0, correct_answer_idx].item()
    base_incorr_prob = last_token_probs[0, incorrect_answer_idx].item()
    base_corr_logit = last_token_logits[0, correct_answer_idx].item()
    base_incorr_logit = last_token_logits[0, incorrect_answer_idx].item()
    base_probs = (base_corr_prob - base_incorr_prob, base_corr_prob, base_incorr_prob)
    base_logs = (base_corr_logit - base_incorr_logit, base_corr_logit, base_incorr_logit)
    top_k_probits = torch.topk(last_token_probs[0], 10)
    top_k_logits = torch.topk(last_token_logits[0], 10)
    if not torch.all(top_k_probits.indices == top_k_logits.indices).item():
        print("CHECK THIS INSTANCE:")
        print(top_k_probits)
        print(top_k_logits)
    top_k_probit_tokens = [
        (a.item(), mt.tokenizer.decode(a), b.item(), c.item())
        for a, b, c in zip(top_k_probits.indices, top_k_probits.values, top_k_logits.values)
    ]
    other_token_probs = []
    if len(answers_t) > 2:
        # get other token probs iff they are not the same as the correct or incorrect answer
        other_token_probs.append(last_token_probs[0, correct_contrast_idx].item())
        other_token_probs.append(last_token_probs[0, incorrect_contrast_idx].item())
    if counterfactual:
        # get the counterfactual instance's scores
        contrast_corr_prob = last_token_probs[1, correct_contrast_idx].item()
        contrast_incorr_prob = last_token_probs[1, incorrect_contrast_idx].item()
        contrast_corr_logit = last_token_logits[1, correct_contrast_idx].item()
        contrast_incorr_logit = last_token_logits[1, incorrect_contrast_idx].item()
        counterfactual_probs = (
            contrast_corr_prob - contrast_incorr_prob,
            contrast_corr_prob,
            contrast_incorr_prob,
        )
        counterfactual_logs = (
            contrast_corr_logit - contrast_incorr_logit,
            contrast_corr_logit,
            contrast_incorr_logit,
        )
        return (base_probs, base_logs, other_token_probs, top_k_probit_tokens), (
            counterfactual_probs,
            counterfactual_logs,
        )
    elif return_top_k:
        return (
            base_probs,
            base_logs,
            top_k_probit_tokens,
            correct_answer_rank,
            incorrect_answer_rank,
            correct_answer_prob,
            incorrect_answer_prob,
            correct_answer_logit,
            incorrect_answer_logit,
        )
    else:
        return base_probs, base_logs, other_token_probs, top_k_probit_tokens
