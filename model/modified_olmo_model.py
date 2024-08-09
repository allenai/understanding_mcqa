from __future__ import annotations

from typing import Optional, Tuple

import torch
from olmo.config import BlockType, ModelConfig
from olmo.model import (
    BufferCache,
    Dropout,
    LayerNorm,
    OLMo,
    OLMoBlock,
    OLMoBlockGroup,
    OLMoSequentialBlock,
)


class ModifiedOLMo(OLMo):

    """
    Re-do initialization of transformer to use modified block class.
    This code is otherwise the same as the implementation in olmo v0.4.0 (https://github.com/allenai/OLMo/blob/v0.4.0/olmo/model.py)
    """

    def __init__(self, config: ModelConfig, init_params: bool = True):
        super().__init__(config, init_params)
        self.__cache = BufferCache()

        self.transformer = torch.nn.ModuleDict(
            dict(
                wte=torch.nn.Embedding(
                    config.embedding_size or config.vocab_size,
                    config.d_model,
                    device=config.init_device,
                ),
                emb_drop=Dropout(config.embedding_dropout),
                ln_f=LayerNorm.build(config),
            )
        )

        blocks = [ModifiedOLMoBlock.build(i, config, self.__cache) for i in range(config.n_layers)]
        if self.config.block_group_size > 1:
            block_groups = [
                OLMoBlockGroup(config, i, blocks[i : i + config.block_group_size])
                for i in range(0, config.n_layers, config.block_group_size)
            ]
            self.transformer.update({"block_groups": torch.nn.ModuleList(block_groups)})
        else:
            self.transformer.update({"blocks": torch.nn.ModuleList(blocks)})

        if not (self.config.alibi or self.config.rope):
            self.transformer.update(
                {
                    "wpe": torch.nn.Embedding(
                        config.max_sequence_length, config.d_model, device=config.init_device
                    )
                }
            )
        if not config.weight_tying:
            self.transformer.update(
                {
                    "ff_out": torch.nn.Linear(
                        config.d_model,
                        config.embedding_size or config.vocab_size,
                        bias=config.include_bias,
                        device=config.init_device,
                    )
                }
            )
        # When `init_device="meta"` FSDP will call `reset_parameters()` to initialize weights.
        if init_params and self.config.init_device != "meta":
            self.reset_parameters()


class ModifiedOLMoBlock(OLMoBlock):

    """
    A modified class for transformer block implementations which implements pseudo-layers to hook and store various hidden states in the attention function.
    Unlike for Llama2, these modifications are needed due to the structure of OLMo layers (which does not save the attention function's various outputs as named layers in the model config).
    Note: we don't update the get_fsdp_wrap_policy() function since we don't use FullyShardedDataParallel in our inference runs.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # add additional pseudo-layers (which have no effect on forward pass)
        self.pseudo_layer_1 = torch.nn.Identity()
        self.pseudo_layer_2 = torch.nn.Identity()

    """
    Update attention function to return various intermediate hidden states.
    This code is otherwise the same as the implementation in olmo v0.4.0 (https://github.com/allenai/OLMo/blob/v0.4.0/olmo/model.py)
    """

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = q.size()  # batch size, sequence length, d_model
        dtype = k.dtype

        # Optionally apply layer norm to keys and queries.
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).to(dtype=dtype)
            k = self.k_norm(k).to(dtype=dtype)

        # Move head forward to be next to the batch dim.
        # shape: (B, nh, T, hs)
        q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        k = k.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        v = v.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)

        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        present = (k, v) if use_cache else None
        query_len, key_len = q.shape[-2], k.shape[-2]  # could be different if layer_past not None

        if self.config.rope:
            # Apply rotary embeddings.
            q, k = self.rotary_emb(q, k)

        if attention_bias is not None:
            # Resize and cast attention bias.
            # The current dtype of the attention bias might not match the dtype that the SDP attn function will
            # run in if AMP is enabled, and this can be a problem if some tokens are masked out due to padding
            # as down-casting the attention bias to the autocast precision will result in -infs, which will
            # cause the SDP attn function to produce NaNs.
            attention_bias = self._cast_attn_bias(
                attention_bias[:, :, key_len - query_len : key_len, :key_len], dtype
            )

        # Get the attention scores.
        # shape: (B, nh, T, hs)
        att = self._scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias,
            dropout_p=0.0 if not self.training else self.config.attention_dropout,
            is_causal=attention_bias is None,
        )

        # Re-assemble all head outputs side-by-side.
        att = att.transpose(1, 2).contiguous().view(B, T, C)

        # Modification: run pseudo-layer to store attention head outputs pre-transformation by attn_out matrix.
        att = self.pseudo_layer_1(att)

        att = self.attn_out(att)

        # Modification: run pseudo-layer to store full attention module output pre-dropout.
        att = self.pseudo_layer_2(att)

        # Apply output projection.
        return att, present

    @classmethod
    def build(cls, layer_id: int, config: ModelConfig, cache: BufferCache) -> ModifiedOLMoBlock:
        if config.block_type == BlockType.sequential:
            return ModifiedOLMoSequentialBlock(layer_id, config, cache)
        elif config.block_type == BlockType.parallel:
            raise NotImplementedError(f"Not implemented with required pseudo-layers")
        elif config.block_type == BlockType.llama:
            raise NotImplementedError(f"Not implemented with required pseudo-layers")
        else:
            raise NotImplementedError(f"Unknown block type: '{config.block_type}'")


class ModifiedOLMoSequentialBlock(OLMoSequentialBlock, ModifiedOLMoBlock):
    """
    Initialize this block to inherit (and thus use in its forward pass) the modified attention function.
    """

    def __init__(self, *args, **kwargs):
        ModifiedOLMoBlock.__init__(self, *args, **kwargs)
        OLMoSequentialBlock.__init__(self, *args, **kwargs)
