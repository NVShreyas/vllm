# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode-context-parallel metadata helpers for MiniMax M3."""

from dataclasses import dataclass

import torch

from vllm.v1.attention.backend import CommonAttentionMetadata


@dataclass(frozen=True)
class MiniMaxM3DCPPrefillSegments:
    """Page-aligned views of a prefill batch for one DCP rank.

    Query rows stay in their original order. Each request is split whenever
    DCP ownership changes, making the global-to-local causal mapping affine
    within every segment.
    """

    query_lens: torch.Tensor
    kv_lens: torch.Tensor
    query_offsets: torch.Tensor
    request_indices: torch.Tensor
    query_starts: torch.Tensor
    owned: torch.Tensor


def _dcp_local_len(
    global_len: int,
    dcp_rank: int,
    dcp_world_size: int,
    interleave_size: int,
) -> int:
    rounds, remainder = divmod(global_len, dcp_world_size * interleave_size)
    return rounds * interleave_size + min(
        max(remainder - dcp_rank * interleave_size, 0), interleave_size
    )


def minimax_m3_dcp_prefill_segments(
    query_start_loc_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    *,
    num_decodes: int,
    dcp_rank: int,
    dcp_world_size: int,
    interleave_size: int,
) -> MiniMaxM3DCPPrefillSegments:
    """Split prefill queries at DCP ownership boundaries.

    For an owned segment, ``query_offsets`` maps its causal rows directly into
    the compressed rank-local KV sequence. For a remote segment, all local KV
    is older than every query, so the offset exposes the complete local prefix.
    """
    if query_start_loc_cpu.device.type != "cpu" or seq_lens_cpu.device.type != "cpu":
        raise ValueError("prefill segment planning requires CPU metadata")
    if query_start_loc_cpu.ndim != 1 or seq_lens_cpu.ndim != 1:
        raise ValueError("query starts and sequence lengths must be one-dimensional")
    if dcp_world_size <= 1 or not 0 <= dcp_rank < dcp_world_size:
        raise ValueError("prefill segmentation requires a valid multi-rank DCP group")
    if interleave_size <= 0:
        raise ValueError("interleave_size must be positive")

    query_lens: list[int] = []
    kv_lens: list[int] = []
    query_offsets: list[int] = []
    request_indices: list[int] = []
    query_starts: list[int] = []
    owned: list[bool] = []

    num_reqs = seq_lens_cpu.shape[0]
    for req_idx in range(num_decodes, num_reqs):
        q_begin = int(query_start_loc_cpu[req_idx])
        q_end = int(query_start_loc_cpu[req_idx + 1])
        query_len = q_end - q_begin
        seq_end = int(seq_lens_cpu[req_idx])
        seq_begin = seq_end - query_len
        if query_len <= 0 or seq_begin < 0:
            raise ValueError("invalid prefill query or sequence length")

        global_start = seq_begin
        token_start = q_begin - int(query_start_loc_cpu[num_decodes])
        while global_start < seq_end:
            boundary = (global_start // interleave_size + 1) * interleave_size
            global_end = min(seq_end, boundary)
            segment_len = global_end - global_start
            segment_owned = global_start // interleave_size % dcp_world_size == dcp_rank
            local_start = _dcp_local_len(
                global_start, dcp_rank, dcp_world_size, interleave_size
            )
            local_end = _dcp_local_len(
                global_end, dcp_rank, dcp_world_size, interleave_size
            )

            query_lens.append(segment_len)
            kv_lens.append(local_end)
            # causal row i admits local columns <= i + offset. Remote
            # segments add no local keys, so row zero must already admit the
            # entire local prefix; later rows remain capped by local_end.
            query_offsets.append(
                local_start if segment_owned else max(local_end - 1, 0)
            )
            request_indices.append(req_idx - num_decodes)
            query_starts.append(token_start)
            owned.append(segment_owned)

            global_start = global_end
            token_start += segment_len

    return MiniMaxM3DCPPrefillSegments(
        query_lens=torch.tensor(query_lens, dtype=torch.int32),
        kv_lens=torch.tensor(kv_lens, dtype=torch.int32),
        query_offsets=torch.tensor(query_offsets, dtype=torch.int32),
        request_indices=torch.tensor(request_indices, dtype=torch.int64),
        query_starts=torch.tensor(query_starts, dtype=torch.int64),
        owned=torch.tensor(owned, dtype=torch.bool),
    )


def minimax_m3_decode_seq_lens(
    common_attn_metadata: CommonAttentionMetadata,
    num_decodes: int,
) -> torch.Tensor:
    """Return the KV lengths represented by this rank's decode block table."""
    seq_lens = common_attn_metadata.dcp_local_seq_lens
    if seq_lens is None:
        seq_lens = common_attn_metadata.seq_lens
    return seq_lens[:num_decodes]
