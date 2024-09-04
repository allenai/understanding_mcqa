import re

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

"""
This file is modified from https://github.com/jmerullo/lm_vector_arithmetic/blob/main/modeling.py
and customized for the models of interest.
"""


class ModelWrapper(nn.Module):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model.eval()
        self.model.activations_ = {}
        self.tokenizer = tokenizer
        self.layer_names = [
            n
            for n, _ in model.named_modules()
            if (
                re.match(
                    r"^(transformer|gpt_neox|model|model.transformer)\.(h|layers|blocks)\.\d+$",
                    n,
                )
            )
        ]
        self.num_layers = len(self.layer_names)
        self.hooks = []

    def layer_decode(self, hidden_states):
        raise Exception("Layer decode has to be implemented!")

    def get_layers(self, tokens, **kwargs):
        with torch.inference_mode():
            # returns a tuple of, e.g., len(40) for 40 model layers, containing tensors of shape (batch_size, input_sequence_length, hidden_state_dim)
            outputs = self.model(input_ids=tokens, output_hidden_states=True, **kwargs)

        # per HF's convention, the final layer in the hidden_states tuple is the output logits *after* LN has been applied. we'll pass this in as an argument to the layer_decode argument to avoid applying LN 2x.
        # returns list of len(num_layers) of tensors of shape (vocab_size, batch_size)
        logits = self.layer_decode(outputs.hidden_states, ln_applied_last_layer=True)

        # convert back to a tensor of shape (num_layers, vocab_size, batch_size)
        return torch.stack(logits)

    def rr_per_layer(self, logits, answer_id, debug=False):
        # reciprocal rank of the answer at each layer
        # logits.shape = (num_layers, vocab_size, batch_size)
        probits = F.softmax(logits, dim=1)

        rrs = []
        # repeat separately for each instance in the batch
        for el in range(probits.shape[2]):
            layerwise_rr = {}
            for i, layer in enumerate(probits[:, :, el]):
                sorted_token_probs = layer.argsort(descending=True)
                # find position of token id in the ranked list of token ids
                rank = float(np.where(sorted_token_probs.cpu().numpy() == answer_id)[0][0])
                # add 1 due to 0-indexing
                layerwise_rr[i] = 1 / (rank + 1)
            rrs.append(layerwise_rr)

        return rrs

    def prob_of_answer_per_layer(self, logits, answer_id, debug=False):
        # returns prob of the answer at each layer
        if logits.shape[2] != 1:
            print("code not designed to work for batch sizes >1")
            breakpoint()
        probits = F.softmax(logits, dim=1)
        probs_of_answer = probits[:, answer_id, :].cpu().detach().squeeze()
        # returns a list of length num_layers
        return probs_of_answer.tolist()

    def log_of_answer_per_layer(self, logits, answer_id, debug=False):
        # returns log of the answer at each layer
        if logits.shape[2] != 1:
            print("code not designed to work for batch sizes >1")
            breakpoint()
        logits_of_answer = logits[:, answer_id, :].cpu().detach().squeeze()
        # returns a list of length num_layers
        return logits_of_answer.tolist()

    def topk_per_layer(self, logits, k=10, log=False):
        # returns top-k tokens at each layer
        if log:
            # keep as logits
            probits = logits
        else:
            # perform Softmax along the vocabulary dimension
            probits = F.softmax(logits, dim=1)
        topk = []
        # for each instance in batch
        for el in range(probits.shape[2]):
            layerwise_topk = {}
            for i, layer in enumerate(probits[:, :, el]):
                top_k_token_ids = layer.argsort(descending=True)[:k]
                top_k_tokens = self.tokenizer.convert_ids_to_tokens(top_k_token_ids)
                layerwise_topk[i] = [
                    (el, layer[k].item()) for (k, el) in zip(top_k_token_ids, top_k_tokens)
                ]
            topk.append(layerwise_topk)
        return topk

    def get_activation(
        self, name, output_only=False, input_only=False, return_all_tokens=False, mlp=False
    ):
        # based on https://github.com/mega002/lm-debugger/blob/01ba7413b3c671af08bc1c315e9cc64f9f4abee2/flask_server/req_res_oop.py#L57
        def hook(module, input, output):
            # input is a tuple ([batch_size, seq_len, hidden_dim],) except in case of mlp, in which case it's [batch_size, seq_len, num_value_vectors]
            # for Olmo: output is a Tensor of shape [batch_size, seq_len, hidden_dim]
            # for Llama: output is a tuple of 3 elements: ([batch_size, seq_len, hidden_dim], None, (something)) and we only want to take the first
            if isinstance(output, tuple):
                output = output[0]
            if input[0].shape != output.shape and not mlp:
                raise Exception("sizing issue-- double check this function")
            elif mlp and (
                input[0].shape[0] != output.shape[0] or input[0].shape[1] != output.shape[1]
            ):
                raise Exception(
                    "sizing issue-- ensure that batch and seq_len dimensions are the same"
                )

            # add activations to dictionary
            # only take the last token representation
            if return_all_tokens:
                if input_only or not output_only:
                    self.model.activations_["in_" + name] = input[0].detach()
                if output_only or not input_only:
                    self.model.activations_["out_" + name] = output.detach()
            else:
                num_tokens = list(input[0].size())[1]
                if input_only or not output_only:
                    self.model.activations_["in_" + name] = input[0][:, num_tokens - 1].detach()
                if output_only or not input_only:
                    self.model.activations_["out_" + name] = output[:, num_tokens - 1].detach()

        return hook


class LLaMAWrapper(ModelWrapper):
    def layer_decode(self, hidden_states, ln_applied_last_layer=False):
        logits = []
        # iterate over each layer
        for i, h in enumerate(hidden_states):
            h = h[:, -1, :]  # (batch, num tokens, embedding size) take the last token
            if i == len(hidden_states) - 1 and ln_applied_last_layer:
                # if doing coarse hidden state fetching using HF's output_hidden_states=True,
                # then the last layer's output is the logits post-LN or RMSNorm. Don't reapply.
                normed = h
            else:
                # in all other cases, apply final RMSNorm
                # compute LN w.r.t the hidden state itself, akin to "early exiting" method
                normed = self.model.model.norm(h)
            l = torch.matmul(self.model.lm_head.weight, normed.T)
            logits.append(l)
        # returns a list of len(num_layers) of tensors of shape (vocab_size, batch_size)
        return logits

    def add_hooks(self, type=None, return_all_tokens=False):
        for i in range(self.num_layers):
            if type == "mlp":
                self.hooks.append(
                    self.model.model.layers[i].mlp.register_forward_hook(
                        self.get_activation(
                            f"mlp_{i}",
                            output_only=True,
                            return_all_tokens=return_all_tokens,
                            mlp=True,
                        )
                    )
                )
            elif type == "attn_heads":
                # get the input to the final weight matrix in the attention layer
                self.hooks.append(
                    self.model.model.layers[i].self_attn.pseudo_layer.register_forward_hook(
                        self.get_activation(
                            f"attn_heads_{i}", input_only=True, return_all_tokens=return_all_tokens
                        )
                    )
                )
            elif type == "attention":
                self.hooks.append(
                    self.model.model.layers[i].self_attn.register_forward_hook(
                        self.get_activation(
                            f"attn_{i}", output_only=True, return_all_tokens=return_all_tokens
                        )
                    )
                )
            else:
                raise Exception("invalid hook type specification")


class OlmoWrapper(ModelWrapper):
    def layer_decode(self, hidden_states, ln_applied_last_layer=False):
        logits = []
        # iterate over each layer
        for i, h in enumerate(hidden_states):
            h = h[:, -1, :]  # (batch, num tokens, embedding size) take the last token
            if i == len(hidden_states) - 1 and ln_applied_last_layer:
                # if doing coarse hidden state fetching using HF's output_hidden_states=True,
                # then the last layer's output is the logits post-LN or RMSNorm. Don't reapply.
                normed = h
            else:
                # in all other cases, apply final layernorm
                normed = self.model.model.transformer.ln_f(h)
            # weights not tied in 7B model
            l = torch.matmul(self.model.model.transformer.ff_out.weight, normed.T)
            logits.append(l)
        # returns a list of len(num_layers) of tensors of shape (vocab_size, batch_size)
        return logits

    def add_hooks(self, type=None, return_all_tokens=False):
        for i in range(self.num_layers):
            if type == "mlp":
                self.hooks.append(
                    self.model.model.transformer.blocks[i].ff_out.register_forward_hook(
                        self.get_activation(
                            f"mlp_{i}",
                            output_only=True,
                            return_all_tokens=return_all_tokens,
                            mlp=True,
                        )
                    )
                )
            elif type == "attn_heads":
                # get the input to the final weight matrix in the attention layer
                # since we are using attn_out to do lots of things, we are adding a pseudo-layer to get the outputs from it
                self.hooks.append(
                    self.model.model.transformer.blocks[i].pseudo_layer_1.register_forward_hook(
                        self.get_activation(
                            f"attn_heads_{i}", input_only=True, return_all_tokens=return_all_tokens
                        )
                    )
                )
            elif type == "attention":
                # since we are using attn_out to do lots of things, we are adding a pseudo-layer to get the inputs to it
                self.hooks.append(
                    self.model.model.transformer.blocks[i].pseudo_layer_2.register_forward_hook(
                        self.get_activation(
                            f"attn_{i}", output_only=True, return_all_tokens=return_all_tokens
                        )
                    )
                )
            else:
                raise Exception("invalid hook type specification")
