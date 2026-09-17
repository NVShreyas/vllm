# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DCP glue kernels for the MiniMax M3 MSA indexer.

The score and attention kernels operate on rank-local sparse blocks. These
small kernels preserve the model's *global* top-k semantics across a DCP group:

* force global init/tail blocks only on the rank that owns each block; and
* merge the per-rank candidates, select the deterministic global top-k, then
  filter/localize the winners for rank-local sparse attention.

One sparse block is one DCP interleave unit for the supported M3 configuration,
so ``global_block = local_block * dcp_world_size + dcp_rank``.
"""

import torch

from vllm.triton_utils import tl, triton


def write_dcp_local_seq_lens(
    global_lens: torch.Tensor,
    output: torch.Tensor,
    *,
    global_offset: int,
    dcp_rank: int,
    dcp_world_size: int,
    interleave_size: int,
    output_block_size: int = 1,
) -> None:
    """Localize global token lengths into caller-owned storage."""
    assert global_lens.ndim == 1
    assert output.shape == global_lens.shape and output.dtype == torch.int32
    _write_dcp_local_seq_lens_kernel[(global_lens.shape[0],)](
        global_lens,
        output,
        GLOBAL_OFFSET=global_offset,
        DCP_RANK=dcp_rank,
        DCP_WORLD_SIZE=dcp_world_size,
        INTERLEAVE_SIZE=interleave_size,
        OUTPUT_BLOCK_SIZE=output_block_size,
    )


@triton.jit
def _write_dcp_local_seq_lens_kernel(
    global_lens,
    output,
    GLOBAL_OFFSET: tl.constexpr,
    DCP_RANK: tl.constexpr,
    DCP_WORLD_SIZE: tl.constexpr,
    INTERLEAVE_SIZE: tl.constexpr,
    OUTPUT_BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    global_len = tl.maximum(tl.load(global_lens + row) + GLOBAL_OFFSET, 0)
    base = global_len // INTERLEAVE_SIZE // DCP_WORLD_SIZE * INTERLEAVE_SIZE
    remainder = global_len - base * DCP_WORLD_SIZE
    remainder = tl.minimum(
        tl.maximum(remainder - DCP_RANK * INTERLEAVE_SIZE, 0),
        INTERLEAVE_SIZE,
    )
    local_len = base + remainder
    output_value = (local_len + OUTPUT_BLOCK_SIZE - 1) // OUTPUT_BLOCK_SIZE
    tl.store(output + row, output_value)


def force_dcp_global_blocks(
    scores: torch.Tensor,
    global_num_valid_pages: torch.Tensor,
    *,
    dcp_rank: int,
    dcp_world_size: int,
    init_blocks: int,
    local_blocks: int,
) -> None:
    """Force model-global init/tail blocks in rank-local score rows."""
    if init_blocks == 0 and local_blocks == 0:
        return
    assert scores.ndim == 2
    assert global_num_valid_pages.shape == (scores.shape[0],)
    num_forced_slots = init_blocks + local_blocks
    _force_dcp_global_blocks_kernel[(scores.shape[0],)](
        scores,
        global_num_valid_pages,
        scores.stride(0),
        scores.shape[1],
        DCP_RANK=dcp_rank,
        DCP_WORLD_SIZE=dcp_world_size,
        INIT_BLOCKS=init_blocks,
        LOCAL_BLOCKS=local_blocks,
        NUM_FORCED_SLOTS=num_forced_slots,
        BLOCK_SIZE=triton.next_power_of_2(num_forced_slots),
    )


@triton.jit
def _force_dcp_global_blocks_kernel(
    scores,
    global_num_valid_pages,
    score_stride0: tl.constexpr,
    num_local_pages,
    DCP_RANK: tl.constexpr,
    DCP_WORLD_SIZE: tl.constexpr,
    INIT_BLOCKS: tl.constexpr,
    LOCAL_BLOCKS: tl.constexpr,
    NUM_FORCED_SLOTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.arange(0, BLOCK_SIZE)
    global_valid = tl.load(global_num_valid_pages + row)

    # The first INIT_BLOCKS slots name global prefix blocks. The remaining
    # slots name the final LOCAL_BLOCKS global blocks for this query token.
    is_init = slot < INIT_BLOCKS
    global_block = tl.where(
        is_init,
        slot,
        global_valid - LOCAL_BLOCKS + (slot - INIT_BLOCKS),
    )
    valid = (
        (slot < NUM_FORCED_SLOTS)
        & (global_block >= 0)
        & (global_block < global_valid)
        & (global_block % DCP_WORLD_SIZE == DCP_RANK)
    )
    local_block = global_block // DCP_WORLD_SIZE
    valid &= local_block < num_local_pages
    tl.store(
        scores + row * score_stride0 + local_block,
        float("inf"),
        mask=valid,
    )


def merge_filter_dcp_topk(
    gathered: torch.Tensor,
    output: torch.Tensor,
    *,
    dcp_rank: int,
    dcp_world_size: int,
) -> None:
    """Select global winners, then write this rank's localized subset.

    ``gathered`` is ``[rows, dcp_world_size * topk, 2]`` with score/global-id
    pairs. ``output`` is ``[rows, topk]`` and is padded with ``-1`` because a
    rank can own anywhere from zero through all global winners.
    """
    assert gathered.ndim == 3 and gathered.shape[2] == 2
    assert output.ndim == 2 and output.shape[0] == gathered.shape[0]
    num_candidates = gathered.shape[1]
    assert num_candidates == dcp_world_size * output.shape[1]
    block_size = triton.next_power_of_2(num_candidates)
    _merge_filter_dcp_topk_kernel[(gathered.shape[0],)](
        gathered,
        output,
        gathered.stride(0),
        gathered.stride(1),
        gathered.stride(2),
        output.stride(0),
        NUM_CANDIDATES=num_candidates,
        TOPK=output.shape[1],
        DCP_RANK=dcp_rank,
        DCP_WORLD_SIZE=dcp_world_size,
        BLOCK_SIZE=block_size,
        num_warps=1,
    )


@triton.jit
def _merge_filter_dcp_topk_kernel(
    gathered,
    output,
    gathered_stride0: tl.constexpr,
    gathered_stride1: tl.constexpr,
    gathered_stride2: tl.constexpr,
    output_stride0: tl.constexpr,
    NUM_CANDIDATES: tl.constexpr,
    TOPK: tl.constexpr,
    DCP_RANK: tl.constexpr,
    DCP_WORLD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK_SIZE)
    candidate_mask = lane < NUM_CANDIDATES
    base = gathered + row * gathered_stride0 + lane * gathered_stride1
    score = tl.load(base, mask=candidate_mask, other=-float("inf"))
    global_id = tl.load(
        base + gathered_stride2,
        mask=candidate_mask,
        other=-1.0,
    ).to(tl.int32)

    # A unique sortable key gives score-descending, global-id-ascending order.
    # Canonicalize NaNs and signed zero before applying the standard monotonic
    # float-to-uint transform. Invalid/padding candidates sort last as key 0.
    score = tl.where(score != score, -float("inf"), score)
    score = tl.where(score == 0.0, 0.0, score)
    score_bits = score.to(tl.uint32, bitcast=True)
    ordered_score = score_bits ^ tl.where(
        score_bits >> 31 != 0,
        0xFFFFFFFF,
        0x80000000,
    )
    key = (ordered_score.to(tl.uint64) << 32) | (~global_id.to(tl.uint32)).to(tl.uint64)
    # sparse_topk_select may leave a nonnegative id in padded candidate slots
    # when a row has fewer than TOPK valid pages. The unified score buffer's
    # -inf sentinel is the source of truth for those unwritten/invalid slots;
    # never allow them to become global winners and address an invalid KV page.
    valid_candidate = candidate_mask & (global_id >= 0) & (score != -float("inf"))
    key = tl.where(valid_candidate, key, 0)
    key = tl.sort(key, descending=True)

    selected = lane < TOPK
    selected_global_id = (~(key & 0xFFFFFFFF).to(tl.uint32)).to(tl.int32)
    owned = selected & (key != 0) & (selected_global_id % DCP_WORLD_SIZE == DCP_RANK)
    local_id = selected_global_id // DCP_WORLD_SIZE
    compact_slot = tl.cumsum(owned.to(tl.int32), axis=0) - 1

    # Every row has TOPK output slots but only the owning rank retains each
    # global winner. Initialize all slots, then compact this rank's winners.
    tl.store(
        output + row * output_stride0 + lane,
        -1,
        mask=lane < TOPK,
    )
    tl.store(
        output + row * output_stride0 + compact_slot,
        local_id,
        mask=owned,
    )
