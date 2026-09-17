# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.models.minimax_m3.nvidia import (
    sparse_attention_msa as sparse_attention_msa_module,
)
from vllm.models.minimax_m3.nvidia.sparse_attention_msa import (
    MiniMaxM3SparseMetadata,
    MiniMaxM3SparseMSADecodeMetadata,
    MiniMaxM3SparseMSAImpl,
)


class _FakeDCPGroup:
    world_size = 2
    rank_in_group = 0

    def __init__(self) -> None:
        self.gather_input: torch.Tensor | None = None
        self.gather_dim: int | None = None

    def all_gather(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        self.gather_input = tensor
        self.gather_dim = dim
        return torch.cat((tensor, tensor), dim=dim)


class _FakeWorkspace:
    def get(self, num_tokens: int):
        return (
            torch.empty((num_tokens, 16, 128), dtype=torch.bfloat16),
            torch.empty((num_tokens, 16), dtype=torch.float32),
        )


def _dcp_impl() -> MiniMaxM3SparseMSAImpl:
    impl = object.__new__(MiniMaxM3SparseMSAImpl)
    impl.num_heads = 8
    impl.head_size = 128
    impl.num_kv_heads = 1
    impl.scale = 0.125
    impl.topk_blocks = 16
    impl.block_size = 128
    impl.kv_cache_dtype = "auto"
    impl.use_fp8_kv = False
    impl.use_cutlass_decode = False
    impl.use_flashinfer_decode = True
    impl.dcp_world_size = 2
    impl.dcp_rank = 0
    impl.need_to_return_lse_for_decode = True
    impl.workspace = _FakeWorkspace()
    return impl


def _metadata(decode_query_len: int = 1) -> MiniMaxM3SparseMetadata:
    num_decodes = 3
    num_decode_tokens = num_decodes * decode_query_len
    decode = MiniMaxM3SparseMSADecodeMetadata(
        seq_lens=torch.tensor([5, 0, 7], dtype=torch.int32),
        block_table=torch.zeros((num_decodes, 1), dtype=torch.int32),
        decode_query_len=decode_query_len,
        query_start_loc=torch.arange(
            0,
            num_decode_tokens + 1,
            decode_query_len,
            dtype=torch.int32,
        ),
        context_seq_lens=torch.tensor([3, 0, 5], dtype=torch.int32),
    )
    return MiniMaxM3SparseMetadata(
        seq_lens=decode.seq_lens,
        max_seq_len=7,
        slot_mapping=torch.arange(num_decode_tokens),
        num_actual_tokens=num_decode_tokens,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        num_prefills=0,
        num_prefill_tokens=0,
        decode=decode,
    )


def test_flashinfer_dcp_gathers_query_and_combines_partial_attention(monkeypatch):
    impl = _dcp_impl()
    metadata = _metadata()
    group = _FakeDCPGroup()
    layer = SimpleNamespace(
        layer_name="model.layers.0.self_attn",
        topk_indices_buffer=torch.arange(3 * 16).view(3, 1, 16),
        _q_scale_float=0.25,
        _k_scale_float=0.5,
        _v_scale_float=0.75,
    )
    query = torch.zeros((3, 8 * 128), dtype=torch.bfloat16)
    query_fp8 = torch.zeros((3, 8 * 128), dtype=torch.float8_e4m3fn)
    output = torch.full_like(query, -1)
    combined = torch.arange(3 * 8 * 128, dtype=torch.float32).view(3, 8, 128)
    combined = combined.to(torch.bfloat16)
    observed: dict[str, Any] = {}

    monkeypatch.setattr(
        sparse_attention_msa_module,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={layer.layer_name: metadata}),
    )
    monkeypatch.setattr(sparse_attention_msa_module, "get_dcp_group", lambda: group)

    def fake_flashinfer(query, kv_cache, topk, block_table, seq_lens, dql, **kwargs):
        observed["query"] = query.clone()
        observed["topk"] = topk.clone()
        observed["seq_lens"] = seq_lens.clone()
        observed["dql"] = dql
        observed["causal"] = kwargs["causal"]
        observed["scales"] = (
            kwargs["q_scale_float"],
            kwargs["k_scale_float"],
            kwargs["v_scale_float"],
        )
        kwargs["out"].fill_(4)
        kwargs["lse_out"].fill_(2)
        return kwargs["out"], kwargs["lse_out"]

    def fake_combine(partial_out, partial_lse, cp_group, **kwargs):
        observed["partial_out"] = partial_out.clone()
        observed["partial_lse"] = partial_lse.clone()
        observed["combine_group"] = cp_group
        observed["lse_base_on_e"] = kwargs["is_lse_base_on_e"]
        return combined

    monkeypatch.setattr(
        sparse_attention_msa_module,
        "msa_flashinfer_sparse_decode",
        fake_flashinfer,
    )
    monkeypatch.setattr(
        sparse_attention_msa_module,
        "cp_lse_ag_out_rs",
        fake_combine,
    )

    result = impl.forward(layer, query, torch.empty(1), output, query_fp8=query_fp8)

    assert result is output
    assert group.gather_dim == 1
    assert group.gather_input is not None
    assert group.gather_input.shape == (3, 8, 128)
    assert observed["query"].shape == (3, 16, 128)
    torch.testing.assert_close(
        observed["topk"], layer.topk_indices_buffer.transpose(0, 1)
    )
    torch.testing.assert_close(observed["seq_lens"], metadata.decode.seq_lens)
    assert observed["dql"] == 1
    assert observed["causal"] is True
    assert observed["scales"] == (0.25, 0.5, 0.75)
    assert observed["combine_group"] is group
    assert observed["lse_base_on_e"] is True

    partial_out = observed["partial_out"]
    partial_lse = observed["partial_lse"]
    assert partial_out.shape == (3, 16, 128)
    assert partial_lse.shape == (3, 16)
    torch.testing.assert_close(partial_out[0], torch.full_like(partial_out[0], 4))
    torch.testing.assert_close(partial_out[1], torch.zeros_like(partial_out[1]))
    torch.testing.assert_close(partial_out[2], torch.full_like(partial_out[2], 4))
    assert torch.isneginf(partial_lse[1]).all()
    torch.testing.assert_close(output.view(3, 8, 128), combined)


@pytest.mark.parametrize("decode_query_len", [2, 3, 4])
def test_flashinfer_dcp_multi_token_merges_context_and_causal_suffix(
    monkeypatch, decode_query_len
):
    impl = _dcp_impl()
    metadata = _metadata(decode_query_len=decode_query_len)
    group = _FakeDCPGroup()
    num_tokens = 3 * decode_query_len
    layer = SimpleNamespace(
        layer_name="model.layers.0.self_attn",
        topk_indices_buffer=torch.zeros((num_tokens, 1, 16), dtype=torch.int32),
    )
    query = torch.zeros((num_tokens, 8 * 128), dtype=torch.bfloat16)
    query_fp8 = torch.zeros((num_tokens, 8 * 128), dtype=torch.float8_e4m3fn)
    current_key = torch.zeros((num_tokens, 128), dtype=torch.bfloat16)
    current_value = torch.zeros_like(current_key)
    output = torch.empty_like(query)
    observed: dict[str, Any] = {}

    monkeypatch.setattr(
        sparse_attention_msa_module,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={layer.layer_name: metadata}),
    )
    monkeypatch.setattr(sparse_attention_msa_module, "get_dcp_group", lambda: group)

    def fake_flashinfer(query, kv_cache, topk, block_table, seq_lens, dql, **kwargs):
        observed["context_dql"] = dql
        observed["context_causal"] = kwargs["causal"]
        observed["context_seq_lens"] = seq_lens.clone()
        kwargs["out"].fill_(2)
        kwargs["lse_out"].fill_(1)
        return kwargs["out"], kwargs["lse_out"]

    def fake_combine(partial_out, partial_lse, cp_group, **kwargs):
        observed["return_lse"] = kwargs["return_lse"]
        return partial_out[:, :8].clone(), partial_lse[:, :8].clone()

    def fake_flash_attn_varlen_func(**kwargs):
        observed["suffix_causal"] = kwargs["causal"]
        observed["suffix_qsl"] = kwargs["cu_seqlens_q"].clone()
        kwargs["out"].fill_(3)
        return kwargs["out"], torch.zeros((8, num_tokens), dtype=torch.float32)

    def fake_merge(out, context_out, context_lse, suffix_out, suffix_lse):
        observed["context_lse_shape"] = context_lse.shape
        observed["suffix_lse_shape"] = suffix_lse.shape
        out.copy_(context_out + suffix_out)

    monkeypatch.setattr(
        sparse_attention_msa_module,
        "msa_flashinfer_sparse_decode",
        fake_flashinfer,
    )
    monkeypatch.setattr(sparse_attention_msa_module, "cp_lse_ag_out_rs", fake_combine)
    monkeypatch.setattr(sparse_attention_msa_module, "merge_attn_states", fake_merge)
    from vllm.v1.attention.backends import fa_utils

    monkeypatch.setattr(
        fa_utils,
        "flash_attn_varlen_func",
        fake_flash_attn_varlen_func,
        raising=False,
    )
    monkeypatch.setattr(
        "vllm.v1.attention.backends.fa_utils.get_flash_attn_version",
        lambda head_size: 3,
    )

    result = impl.forward(
        layer,
        query,
        torch.empty(1),
        output,
        query_fp8=query_fp8,
        current_key=current_key,
        current_value=current_value,
    )

    assert result is output
    assert observed["context_dql"] == decode_query_len
    assert observed["context_causal"] is False
    assert observed["suffix_causal"] is True
    assert observed["return_lse"] is True
    torch.testing.assert_close(
        observed["context_seq_lens"], metadata.decode.context_seq_lens
    )
    torch.testing.assert_close(observed["suffix_qsl"], metadata.decode.query_start_loc)
    assert observed["context_lse_shape"] == (8, num_tokens)
    assert observed["suffix_lse_shape"] == (8, num_tokens)
    expected = torch.full_like(output, 5)
    expected[decode_query_len : 2 * decode_query_len].fill_(3)
    torch.testing.assert_close(output, expected)
