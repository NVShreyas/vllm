# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.minimax_m3.nvidia.ops.dcp_topk import (
    force_dcp_global_blocks,
    merge_filter_dcp_topk,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test requires CUDA"
)


def _stable_global_topk(scores: torch.Tensor, topk: int) -> list[list[int]]:
    result = []
    for row in scores.cpu().tolist():
        result.append(
            sorted(range(len(row)), key=lambda block: (-row[block], block))[:topk]
        )
    return result


def test_minimax_m3_dcp_force_and_merge_matches_global_reference() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda")
    world = 2
    topk = 16
    rows = 4
    global_valid = torch.tensor([17, 23, 31, 40], dtype=torch.int32, device=device)
    max_global_pages = int(global_valid.max().item())

    global_scores = torch.randn(rows, max_global_pages, device=device)
    # Exercise deterministic tie-breaking independently of the forced blocks.
    global_scores[:, 5:8] = 3.0
    valid_mask = (
        torch.arange(max_global_pages, device=device)[None, :] < global_valid[:, None]
    )
    global_scores.masked_fill_(~valid_mask, -float("inf"))

    packed_by_rank = []
    local_scores_by_rank = []
    for rank in range(world):
        local_scores = global_scores[:, rank::world].contiguous()
        local_valid = (global_valid + (world - 1 - rank)) // world
        local_mask = (
            torch.arange(local_scores.shape[1], device=device)[None, :]
            < (local_valid[:, None])
        )
        local_scores.masked_fill_(~local_mask, -float("inf"))
        force_dcp_global_blocks(
            local_scores,
            global_valid,
            dcp_rank=rank,
            dcp_world_size=world,
            init_blocks=2,
            local_blocks=1,
        )
        local_scores_by_rank.append(local_scores)

        packed = torch.full((rows, topk, 2), -1.0, dtype=torch.float32, device=device)
        for row in range(rows):
            valid_count = int(local_valid[row].item())
            count = min(topk, valid_count)
            local_ids = torch.argsort(
                local_scores[row, :valid_count],
                descending=True,
                stable=True,
            )[:count]
            packed[row, :count, 0] = local_scores[row, local_ids]
            packed[row, :count, 1] = (local_ids * world + rank).float()
        packed_by_rank.append(packed)

    forced_global_scores = global_scores.clone()
    for row, valid in enumerate(global_valid.cpu().tolist()):
        forced_global_scores[row, : min(2, valid)] = float("inf")
        forced_global_scores[row, max(0, valid - 1) : valid] = float("inf")
    expected_global = _stable_global_topk(forced_global_scores, topk)

    gathered = torch.cat(packed_by_rank, dim=1).contiguous()
    for rank in range(world):
        actual = torch.empty((rows, topk), dtype=torch.int32, device=device)
        merge_filter_dcp_topk(
            gathered,
            actual,
            dcp_rank=rank,
            dcp_world_size=world,
        )
        expected = []
        for winners in expected_global:
            owned = [block // world for block in winners if block % world == rank]
            expected.append(owned + [-1] * (topk - len(owned)))
        torch.testing.assert_close(
            actual.cpu(), torch.tensor(expected, dtype=torch.int32)
        )

    # The force kernel must alter only the owner-local representations.
    for row, valid in enumerate(global_valid.cpu().tolist()):
        for global_block in [0, 1, valid - 1]:
            rank = global_block % world
            local_block = global_block // world
            assert torch.isinf(local_scores_by_rank[rank][row, local_block])


def test_minimax_m3_dcp_merge_covers_every_local_winner_count() -> None:
    device = torch.device("cuda")
    world = 2
    topk = 16
    rows = topk + 1
    gathered = torch.empty((rows, world * topk, 2), device=device)
    expected_by_rank: list[list[list[int]]] = [[], []]

    for rank0_winners in range(topk + 1):
        rank1_winners = topk - rank0_winners
        winner_ids = [2 * i for i in range(rank0_winners)] + [
            2 * i + 1 for i in range(rank1_winners)
        ]
        loser_ids = [
            block for block in range(world * topk) if block not in set(winner_ids)
        ]
        ordered_ids = winner_ids + loser_ids
        for position, global_id in enumerate(ordered_ids):
            gathered[rank0_winners, position, 0] = float(world * topk - position)
            gathered[rank0_winners, position, 1] = float(global_id)
        for rank in range(world):
            local = [block // world for block in winner_ids if block % world == rank]
            expected_by_rank[rank].append(local + [-1] * (topk - len(local)))

    for rank in range(world):
        actual = torch.empty((rows, topk), dtype=torch.int32, device=device)
        merge_filter_dcp_topk(
            gathered,
            actual,
            dcp_rank=rank,
            dcp_world_size=world,
        )
        torch.testing.assert_close(
            actual.cpu(), torch.tensor(expected_by_rank[rank], dtype=torch.int32)
        )


def test_minimax_m3_dcp_merge_rejects_sentinel_candidates() -> None:
    """Rows shorter than top-k must not retain padded/stale block ids."""
    device = torch.device("cuda")
    world = 2
    topk = 16
    gathered = torch.full((2, world * topk, 2), -1.0, device=device)

    # Each row has one real global block. Model the stale nonnegative ids that
    # sparse_topk_select can leave behind, but retain the unified buffer's -inf
    # sentinel score for every invalid slot.
    gathered[:, :, 0] = -float("inf")
    gathered[:, :, 1] = torch.arange(world * topk, device=device)
    gathered[0, 0] = torch.tensor([3.0, 0.0], device=device)
    gathered[1, topk] = torch.tensor([4.0, 1.0], device=device)

    expected = (
        [[0] + [-1] * (topk - 1), [-1] * topk],
        [[-1] * topk, [0] + [-1] * (topk - 1)],
    )
    for rank in range(world):
        actual = torch.empty((2, topk), dtype=torch.int32, device=device)
        merge_filter_dcp_topk(
            gathered,
            actual,
            dcp_rank=rank,
            dcp_world_size=world,
        )
        torch.testing.assert_close(
            actual.cpu(), torch.tensor(expected[rank], dtype=torch.int32)
        )
