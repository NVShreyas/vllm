# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode-context-parallel metadata helpers for MiniMax M3."""

import torch

from vllm.v1.attention.backend import CommonAttentionMetadata


def minimax_m3_decode_seq_lens(
    common_attn_metadata: CommonAttentionMetadata,
    num_decodes: int,
) -> torch.Tensor:
    """Return the KV lengths represented by this rank's decode block table."""
    seq_lens = common_attn_metadata.dcp_local_seq_lens
    if seq_lens is None:
        seq_lens = common_attn_metadata.seq_lens
    return seq_lens[:num_decodes]
