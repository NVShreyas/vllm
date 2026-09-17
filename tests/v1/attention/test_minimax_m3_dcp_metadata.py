# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from vllm.models.minimax_m3.common.dcp import minimax_m3_dcp_prefill_segments
from vllm.models.minimax_m3.common.indexer import (
    MiniMaxM3IndexerTritonMetadataBuilder,
)
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparseMetadataBuilder,
)
from vllm.models.minimax_m3.nvidia import (
    sparse_attention_msa as sparse_attention_msa_module,
)
from vllm.models.minimax_m3.nvidia.indexer_msa import (
    MAX_K_TILES,
    MiniMaxM3IndexerMSAMetadataBuilder,
)
from vllm.models.minimax_m3.nvidia.sparse_attention_msa import (
    MiniMaxM3SparseMSAMetadataBuilder,
)

GLOBAL_SEQ_LENS = [1, 127, 128, 129, 255, 256, 257]


@pytest.mark.parametrize(
    ("dcp_rank", "expected_kv_lens", "expected_offsets", "expected_owned"),
    [
        pytest.param(
            0,
            [128, 128, 192, 192],
            [64, 127, 128, 128],
            [True, False, True, True],
            id="rank-0",
        ),
        pytest.param(
            1,
            [0, 128, 128, 128],
            [0, 0, 127, 127],
            [False, True, False, False],
            id="rank-1",
        ),
    ],
)
def test_dcp_prefill_segments_preserve_query_order_and_local_causality(
    dcp_rank, expected_kv_lens, expected_offsets, expected_owned
):
    # One decode row followed by two prefills. The first prefill crosses three
    # ownership regions from global position 64 through 320; the second stays
    # within rank 0's global block [256, 384).
    segments = minimax_m3_dcp_prefill_segments(
        torch.tensor([0, 1, 257, 321], dtype=torch.int32),
        torch.tensor([17, 320, 320], dtype=torch.int32),
        num_decodes=1,
        dcp_rank=dcp_rank,
        dcp_world_size=2,
        interleave_size=128,
    )

    assert segments.query_lens.tolist() == [64, 128, 64, 64]
    assert segments.kv_lens.tolist() == expected_kv_lens
    assert segments.query_offsets.tolist() == expected_offsets
    assert segments.request_indices.tolist() == [0, 0, 0, 1]
    assert segments.query_starts.tolist() == [0, 64, 192, 256]
    assert segments.owned.tolist() == expected_owned
    assert int(segments.query_lens.sum()) == 320


def _common_decode_metadata(local_seq_lens: list[int]):
    num_reqs = len(GLOBAL_SEQ_LENS)
    common = create_common_attn_metadata(
        BatchSpec(GLOBAL_SEQ_LENS, [1] * num_reqs),
        block_size=128,
        device=torch.device("cpu"),
        arange_block_indices=True,
    )
    common.positions = torch.tensor(GLOBAL_SEQ_LENS, dtype=torch.int64) - 1
    common.dcp_local_seq_lens = torch.tensor(local_seq_lens, dtype=torch.int32)
    common.dcp_local_seq_lens_cpu = common.dcp_local_seq_lens.cpu()
    return common


def _triton_indexer_builder(num_reqs: int):
    builder = object.__new__(MiniMaxM3IndexerTritonMetadataBuilder)
    builder.reorder_batch_threshold = 1
    builder.context_len_buffer = torch.empty(num_reqs, dtype=torch.int32)
    builder.max_decode_query_len = 1
    return builder


def _sparse_builder(num_reqs: int):
    builder = object.__new__(MiniMaxM3SparseMetadataBuilder)
    builder.reorder_batch_threshold = 1
    builder.context_len_buffer = torch.empty(num_reqs, dtype=torch.int32)
    builder.use_aiter_sparse_pa = False
    builder.page16_slot_mapping_buffer = None
    return builder


def _msa_indexer_builder(num_reqs: int, dcp_rank: int):
    builder = object.__new__(MiniMaxM3IndexerMSAMetadataBuilder)
    builder.reorder_batch_threshold = 1
    builder.context_len_buffer = torch.empty(num_reqs, dtype=torch.int32)
    builder.num_valid_pages_buffer = torch.empty(num_reqs, dtype=torch.int32)
    builder.global_num_valid_pages_buffer = torch.empty(num_reqs, dtype=torch.int32)
    builder.unified_scores_buffer = torch.empty(num_reqs, 1, MAX_K_TILES)
    builder.dcp_packed_candidates_buffer = torch.empty(num_reqs, 16, 2)
    builder.dcp_world_size = 2
    builder.dcp_rank = dcp_rank
    builder.cp_kv_cache_interleave_size = 128
    builder.max_decode_query_len = 1
    builder.num_index_heads = 1
    return builder


def _msa_sparse_builder(num_reqs: int, decode_backend: str, dcp_rank: int = 0):
    builder = object.__new__(MiniMaxM3SparseMSAMetadataBuilder)
    builder.reorder_batch_threshold = 1
    builder.context_len_buffer = torch.empty(num_reqs, dtype=torch.int32)
    builder.use_aiter_sparse_pa = False
    builder.page16_slot_mapping_buffer = None
    builder.decode_backend = decode_backend
    builder.num_q_heads = 16
    builder.num_kv_heads = 1
    builder.topk_blocks = 16
    builder.kv_cache_dtype = "fp8_e4m3"
    builder.msa_cutlass_plan_cache = object()
    builder.dcp_world_size = 2
    builder.dcp_rank = dcp_rank
    builder.cp_kv_cache_interleave_size = 128
    builder.dcp_context_seq_lens_buffer = torch.empty(num_reqs, dtype=torch.int32)
    return builder


@pytest.mark.parametrize(
    (
        "dcp_rank",
        "causal_indices",
        "causal_kv_lens",
        "remote_indices",
        "remote_kv_lens",
    ),
    [
        pytest.param(
            0,
            [*range(128), *range(256, 320)],
            [128, 192],
            [*range(128, 256)],
            [128],
            id="rank-0",
        ),
        pytest.param(
            1,
            [*range(128, 256)],
            [128],
            [*range(128), *range(256, 320)],
            [0, 128],
            id="rank-1",
        ),
    ],
)
def test_msa_attention_segments_dcp_prefill_by_ownership(
    dcp_rank,
    causal_indices,
    causal_kv_lens,
    remote_indices,
    remote_kv_lens,
    monkeypatch,
):
    common = create_common_attn_metadata(
        BatchSpec([320], [320]),
        block_size=128,
        device=torch.device("cpu"),
        arange_block_indices=True,
    )
    monkeypatch.setattr(
        sparse_attention_msa_module,
        "should_prepare_decode_metadata",
        lambda *args, **kwargs: False,
    )

    metadata = _msa_sparse_builder(1, "flashinfer", dcp_rank).build(0, common)
    assert metadata.prefill is not None
    groups = {group.causal: group for group in metadata.prefill.dcp_groups}
    causal = groups[True]
    remote = groups[False]
    assert causal.query_indices.tolist() == causal_indices
    assert causal.seq_lens.tolist() == causal_kv_lens
    assert remote.query_indices.tolist() == remote_indices
    assert remote.seq_lens.tolist() == remote_kv_lens
    assert causal.cu_seqlens_q.dtype == torch.int32
    assert causal.cu_seqlens_k.dtype == torch.int32
    assert remote.cu_seqlens_q.dtype == torch.int32
    assert remote.cu_seqlens_k.dtype == torch.int32
    assert causal.total_kv_blocks == sum(
        (length + 127) // 128 for length in causal_kv_lens
    )
    assert remote.total_kv_blocks == sum(
        (length + 127) // 128 for length in remote_kv_lens
    )


def _cpu_write_dcp_local_seq_lens(
    global_lens,
    output,
    *,
    global_offset,
    dcp_rank,
    dcp_world_size,
    interleave_size,
    output_block_size=1,
):
    global_lens = (global_lens + global_offset).clamp_min(0)
    base = global_lens // interleave_size // dcp_world_size * interleave_size
    remainder = global_lens - base * dcp_world_size
    remainder = (remainder - dcp_rank * interleave_size).clamp(0, interleave_size)
    local_lens = base + remainder
    output.copy_((local_lens + output_block_size - 1) // output_block_size)


@pytest.mark.parametrize(
    ("dcp_rank", "local_seq_lens"),
    [
        pytest.param(0, [1, 127, 128, 128, 128, 128, 129], id="dcp-rank-0"),
        pytest.param(1, [0, 0, 0, 1, 127, 128, 128], id="dcp-rank-1"),
    ],
)
def test_only_msa_indexer_uses_dcp_local_seq_lens(
    dcp_rank, local_seq_lens, monkeypatch
):
    common = _common_decode_metadata(local_seq_lens)
    num_reqs = len(local_seq_lens)

    triton_indexer = _triton_indexer_builder(num_reqs).build(0, common)
    sparse_attention = _sparse_builder(num_reqs).build(0, common)
    monkeypatch.setattr(
        "vllm.models.minimax_m3.nvidia.indexer_msa.write_dcp_local_seq_lens",
        _cpu_write_dcp_local_seq_lens,
    )
    msa_indexer = _msa_indexer_builder(num_reqs, dcp_rank).build(0, common)

    expected_global = torch.tensor(GLOBAL_SEQ_LENS, dtype=torch.int32)
    expected_local = torch.tensor(local_seq_lens, dtype=torch.int32)
    assert triton_indexer.decode is not None
    assert sparse_attention.decode is not None
    assert msa_indexer.decode is not None
    torch.testing.assert_close(triton_indexer.decode.seq_lens, expected_global)
    torch.testing.assert_close(sparse_attention.decode.seq_lens, expected_global)
    torch.testing.assert_close(msa_indexer.decode.seq_lens, expected_local)
    assert triton_indexer.decode.max_seq_len == max(GLOBAL_SEQ_LENS)
    assert msa_indexer.decode.max_seq_len == max(GLOBAL_SEQ_LENS)

    expected_pages = (expected_local + 127) // 128
    torch.testing.assert_close(msa_indexer.topk_num_valid_pages, expected_pages)
    expected_global_pages = (expected_global + 127) // 128
    torch.testing.assert_close(
        msa_indexer.topk_num_global_valid_pages, expected_global_pages
    )


@pytest.mark.parametrize(
    ("decode_backend", "uses_local_lengths"),
    [
        pytest.param("flashinfer", True, id="flashinfer"),
        pytest.param("cutlass", False, id="cutlass"),
        pytest.param("triton", False, id="triton"),
    ],
)
def test_only_flashinfer_attention_uses_dcp_local_seq_lens(
    decode_backend, uses_local_lengths, monkeypatch
):
    local_seq_lens = [1, 127, 128, 128, 128, 128, 129]
    common = _common_decode_metadata(local_seq_lens)
    num_reqs = len(local_seq_lens)
    monkeypatch.setattr(
        sparse_attention_msa_module,
        "should_prepare_decode_metadata",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        sparse_attention_msa_module,
        "write_dcp_local_seq_lens",
        _cpu_write_dcp_local_seq_lens,
    )

    metadata = _msa_sparse_builder(num_reqs, decode_backend).build(0, common)

    assert metadata.decode is not None
    expected = local_seq_lens if uses_local_lengths else GLOBAL_SEQ_LENS
    torch.testing.assert_close(
        metadata.decode.seq_lens, torch.tensor(expected, dtype=torch.int32)
    )
