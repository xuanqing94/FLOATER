# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import time
import pdb
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from fairseq import utils
from fairseq.modules.flow_funcs import *


class FlowAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        self_attention=False,
        encoder_decoder_attention=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        assert self_attention or encoder_decoder_attention
        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        self.in_proj_weights = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.onnx_trace = False

        self.enable_torch_version = False
        if hasattr(F, "multi_head_attention_forward"):
            self.enable_torch_version = True
        else:
            self.enable_torch_version = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weights)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(
        self,
        query,
        key,
        value,
        bias_q,
        bias_k,
        bias_v,
        static_kv=False,
        key_padding_mask=None,
        incremental_state=None,
        attn_mask=None,
    ):
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            bias_q: tensor of shape `(tgt_len, embed_dim)`
            bias_{k/v}: tensor of shape `(src_len, embed_dim)`
        """
        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        assert embed_dim == self.embed_dim
        
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None
        
        if self.self_attention:
            # We know that src_len == tgt_len, so run the flow jointly between K/Q/V
            # proj_weights: [tgt_len, 3 * embed_dim, embed_dim]
            # query: [tgt_len, batch, embed_dim]
            # output: [tgt_len, batch, 3 * embed_dim]
            # q/k/v: [tgt_len, batch, embed_dim]
            assert tgt_len == src_len
            q, k, v = F.linear(query, self.in_proj_weights).chunk(3, dim=-1)
        elif self.encoder_decoder_attention:
            # In this case, src_len != tgt_len (usually), run the flow separately between Q and K/V
            q = F.linear(query, self.in_proj_weights[:embed_dim])
            if key is not None:
                k = F.linear(key, self.in_proj_weights[embed_dim : 2 * embed_dim])
            else:
                k = None
            if value is not None:
                v = F.linear(value, self.in_proj_weights[2 * embed_dim :])
            else:
                v = None
        # Add bias to them
        q += bias_q.unsqueeze(dim=1)
        q *= self.scaling
        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        if k is not None:
            k += bias_k.unsqueeze(dim=1)
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v += bias_v.unsqueeze(dim=1)
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz*num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                prev_key = saved_state["prev_key"].view(
                    bsz * self.num_heads, -1, self.head_dim
                )
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            if "prev_value" in saved_state:
                prev_value = saved_state["prev_value"].view(
                    bsz * self.num_heads, -1, self.head_dim
                )
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)
            if (
                "prev_key_padding_mask" in saved_state
                and saved_state["prev_key_padding_mask"] is not None
            ):
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
                if static_kv:
                    key_padding_mask = prev_key_padding_mask
                else:
                    key_padding_mask = torch.cat(
                        (prev_key_padding_mask, key_padding_mask), dim=1
                    )
            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask

            self._set_input_buffer(incremental_state, saved_state)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.shape == torch.Size([]):
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        #assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_float = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(
            attn_weights_float.type_as(attn_weights),
            p=self.dropout,
            training=self.training,
        )
        attn = torch.bmm(attn_probs, v)
        #assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        return attn, attn_weights

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                if input_buffer[k] is not None:
                    input_buffer[k] = input_buffer[k].index_select(0, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(self, incremental_state, "attn_state") or {}

    def _set_input_buffer(self, incremental_state, buffer):
        utils.set_incremental_state(self, incremental_state, "attn_state", buffer)


if __name__ == "__main__":
    flow_attn = FlowAttention(128, 8, encoder_decoder_attention=True)
    q = torch.randn(13, 3, 128)
    k = torch.randn(9, 3, 128)
    v = torch.randn(9, 3, 128)
    out, _ = flow_attn(q, k, v)
    print(out.size() == q.size())
