# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
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


def _msa_indexer_builder(num_reqs: int):
    builder = object.__new__(MiniMaxM3IndexerMSAMetadataBuilder)
    builder.reorder_batch_threshold = 1
    builder.context_len_buffer = torch.empty(num_reqs, dtype=torch.int32)
    builder.num_valid_pages_buffer = torch.empty(num_reqs, dtype=torch.int32)
    builder.unified_scores_buffer = torch.empty(num_reqs, 1, MAX_K_TILES)
    builder.max_decode_query_len = 1
    builder.num_index_heads = 1
    return builder


def _msa_sparse_builder(num_reqs: int, decode_backend: str):
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
    return builder


@pytest.mark.parametrize(
    "local_seq_lens",
    [
        pytest.param([1, 127, 128, 128, 128, 128, 129], id="dcp-rank-0"),
        pytest.param([0, 0, 0, 1, 127, 128, 128], id="dcp-rank-1"),
    ],
)
def test_only_msa_indexer_uses_dcp_local_seq_lens(local_seq_lens):
    common = _common_decode_metadata(local_seq_lens)
    num_reqs = len(local_seq_lens)

    triton_indexer = _triton_indexer_builder(num_reqs).build(0, common)
    sparse_attention = _sparse_builder(num_reqs).build(0, common)
    msa_indexer = _msa_indexer_builder(num_reqs).build(0, common)

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

    metadata = _msa_sparse_builder(num_reqs, decode_backend).build(0, common)

    assert metadata.decode is not None
    expected = local_seq_lens if uses_local_lengths else GLOBAL_SEQ_LENS
    torch.testing.assert_close(
        metadata.decode.seq_lens, torch.tensor(expected, dtype=torch.int32)
    )
