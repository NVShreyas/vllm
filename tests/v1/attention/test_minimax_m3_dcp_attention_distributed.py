# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory, get_dcp_group
from vllm.distributed.parallel_state import (
    graph_capture,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.models.minimax_m3.nvidia import (
    sparse_attention_msa as sparse_attention_msa_module,
)
from vllm.models.minimax_m3.nvidia.sparse_attention_msa import (
    MiniMaxM3SparseMetadata,
    MiniMaxM3SparseMSADecodeMetadata,
    MiniMaxM3SparseMSAImpl,
    MSASparseAttentionWorkspace,
)
from vllm.platforms import current_platform
from vllm.v1.worker.workspace import init_workspace_manager


class _TorchDCPGroup:
    def __init__(self) -> None:
        self.world_size = dist.get_world_size()
        self.rank_in_group = dist.get_rank()
        self.all_gather_calls = 0
        self.reduce_scatter_calls = 0
        self.all_reduce_calls = 0

    def all_gather(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        self.all_gather_calls += 1
        gathered = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=dim)

    def reduce_scatter(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        self.reduce_scatter_calls += 1
        chunks = [chunk.contiguous() for chunk in tensor.chunk(self.world_size, dim)]
        output = torch.empty_like(chunks[self.rank_in_group])
        dist.reduce_scatter(output, chunks)
        return output

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        self.all_reduce_calls += 1
        dist.all_reduce(tensor)
        return tensor


class _StableWorkspace:
    def __init__(self, max_tokens: int, device: torch.device) -> None:
        self.context_out = torch.empty(
            (max_tokens, 16, 128), dtype=torch.bfloat16, device=device
        )
        self.context_lse = torch.empty(
            (max_tokens, 16), dtype=torch.float32, device=device
        )

    def get(self, num_tokens: int):
        return (
            self.context_out[:num_tokens],
            self.context_lse[:num_tokens],
        )


def _build_caches(
    rank: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    page_counts = [16, 1, 24]
    generator = torch.Generator(device="cpu").manual_seed(11)
    full_pages = sum(page_counts)
    full_cache = torch.randn(
        (full_pages, 1, 128, 256), generator=generator, dtype=torch.float32
    ).mul_(0.25)
    full_cache = full_cache.to(torch.float8_e4m3fn).to(device)

    full_table = torch.zeros((len(page_counts), max(page_counts)), dtype=torch.int32)
    local_table = torch.zeros_like(full_table)
    local_pages: list[torch.Tensor] = []
    full_offset = 0
    local_offset = 0
    for request, count in enumerate(page_counts):
        full_table[request, :count] = torch.arange(
            full_offset, full_offset + count, dtype=torch.int32
        )
        owned = list(range(rank, count, 2))
        for local_id, global_id in enumerate(owned):
            local_table[request, local_id] = local_offset
            local_pages.append(full_cache[full_offset + global_id])
            local_offset += 1
        full_offset += count

    global_winners = [
        list(range(16)),
        [0],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 1, 3, 5, 7],
    ]
    global_topk = torch.full((1, 3, 16), -1, dtype=torch.int32)
    local_topk = torch.full((3, 1, 16), -1, dtype=torch.int32)
    for request, winners in enumerate(global_winners):
        global_topk[0, request, : len(winners)] = torch.tensor(winners)
        local_winners = [block // 2 for block in winners if block % 2 == rank]
        local_topk[request, 0, : len(local_winners)] = torch.tensor(local_winners)

    global_seq_lens = torch.tensor([2048, 128, 3072], dtype=torch.int32)
    local_seq_lens = torch.tensor(
        [1024, 128 if rank == 0 else 0, 1536], dtype=torch.int32
    )
    return (
        full_cache,
        torch.stack(local_pages),
        full_table.to(device),
        local_table.to(device),
        global_topk.to(device),
        local_topk.to(device),
        global_seq_lens.to(device),
        local_seq_lens.to(device),
    )


def _dcp_impl(
    rank: int, device: torch.device, max_tokens: int
) -> MiniMaxM3SparseMSAImpl:
    impl = object.__new__(MiniMaxM3SparseMSAImpl)
    impl.num_heads = 8
    impl.head_size = 128
    impl.num_kv_heads = 1
    impl.scale = 128**-0.5
    impl.topk_blocks = 16
    impl.block_size = 128
    impl.kv_cache_dtype = "fp8_e4m3"
    impl.kv_cache_fp8_dtype = torch.float8_e4m3fn
    impl.use_fp8_kv = True
    impl.use_cutlass_decode = False
    impl.use_flashinfer_decode = True
    impl.dcp_world_size = 2
    impl.dcp_rank = rank
    impl.need_to_return_lse_for_decode = True
    impl.workspace = _StableWorkspace(max_tokens, device)
    return impl


@pytest.mark.skipif(
    not current_platform.is_cuda() or int(os.getenv("WORLD_SIZE", "1")) != 2,
    reason="requires torchrun with two CUDA ranks",
)
@pytest.mark.parametrize("decode_query_len", [2, 3, 4])
def test_minimax_m3_flashinfer_dcp2_noncausal_context_matches_q1_reference(
    decode_query_len,
):
    """Validate the one-call DCP context path independently of dense suffix FA."""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    group = _TorchDCPGroup()

    try:
        (
            full_cache,
            local_cache,
            full_table,
            local_table,
            global_topk,
            local_topk,
            global_seq_lens,
            local_seq_lens,
        ) = _build_caches(rank, device)

        generator = torch.Generator(device="cpu").manual_seed(37)
        num_tokens = 3 * decode_query_len
        global_query = torch.randn(
            (num_tokens, 16, 128), generator=generator, dtype=torch.float32
        ).mul_(0.25)
        global_query = global_query.to(torch.float8_e4m3fn).to(device)
        local_query = global_query[:, rank * 8 : (rank + 1) * 8].contiguous()
        gathered_query = group.all_gather(local_query, dim=1)
        expanded_global_topk = global_topk.repeat_interleave(decode_query_len, dim=1)
        expanded_local_topk = local_topk.repeat_interleave(decode_query_len, dim=0)

        partial_out = torch.empty(
            (num_tokens, 16, 128), dtype=torch.bfloat16, device=device
        )
        partial_lse = torch.empty((num_tokens, 16), dtype=torch.float32, device=device)
        sparse_attention_msa_module.msa_flashinfer_sparse_decode(
            gathered_query,
            local_cache,
            expanded_local_topk.transpose(0, 1),
            local_table,
            local_seq_lens,
            decode_query_len,
            scale=128**-0.5,
            causal=False,
            out=partial_out,
            lse_out=partial_lse,
        )
        empty_shards = local_seq_lens == 0
        empty_tokens = empty_shards.repeat_interleave(decode_query_len)
        partial_out.masked_fill_(empty_tokens[:, None, None], 0)
        partial_lse.masked_fill_(empty_tokens[:, None], float("-inf"))
        combined = sparse_attention_msa_module.cp_lse_ag_out_rs(
            partial_out,
            partial_lse,
            group,
            return_lse=True,
            is_lse_base_on_e=True,
        )
        assert isinstance(combined, tuple)
        combined_out, combined_lse = combined

        reference_out = torch.empty(
            (num_tokens, 16, 128), dtype=torch.bfloat16, device=device
        )
        reference_lse = torch.empty(
            (num_tokens, 16), dtype=torch.float32, device=device
        )
        for query_offset in range(decode_query_len):
            step_out = torch.empty((3, 16, 128), dtype=torch.bfloat16, device=device)
            step_lse = torch.empty((3, 16), dtype=torch.float32, device=device)
            sparse_attention_msa_module.msa_flashinfer_sparse_decode(
                global_query[query_offset::decode_query_len].contiguous(),
                full_cache,
                expanded_global_topk[:, query_offset::decode_query_len].contiguous(),
                full_table,
                global_seq_lens,
                1,
                scale=128**-0.5,
                out=step_out,
                lse_out=step_lse,
            )
            reference_out[query_offset::decode_query_len].copy_(step_out)
            reference_lse[query_offset::decode_query_len].copy_(step_lse)

        expected = reference_out[:, rank * 8 : (rank + 1) * 8]
        expected_lse = reference_lse[:, rank * 8 : (rank + 1) * 8]
        torch.testing.assert_close(combined_out, expected, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(combined_lse, expected_lse, rtol=2e-2, atol=2e-2)
        assert group.all_gather_calls == 2
        assert group.reduce_scatter_calls == 1
        assert group.all_reduce_calls == 0
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not current_platform.is_cuda() or int(os.getenv("WORLD_SIZE", "1")) != 2,
    reason="requires torchrun with two CUDA ranks",
)
@pytest.mark.parametrize("decode_query_len", [1, 2, 3, 4])
def test_minimax_m3_flashinfer_dcp2_matches_unsharded_attention(
    monkeypatch, decode_query_len
):
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    group = _TorchDCPGroup()

    try:
        (
            full_cache,
            local_cache,
            full_table,
            local_table,
            global_topk,
            local_topk,
            global_seq_lens,
            local_seq_lens,
        ) = _build_caches(rank, device)

        generator = torch.Generator(device="cpu").manual_seed(29)
        num_tokens = 3 * decode_query_len
        global_query = torch.randn(
            (num_tokens, 16, 128), generator=generator, dtype=torch.float32
        ).mul_(0.25)
        global_query = global_query.to(torch.float8_e4m3fn).to(device)
        local_query = global_query[:, rank * 8 : (rank + 1) * 8].contiguous()
        expanded_global_topk = global_topk.repeat_interleave(decode_query_len, dim=1)
        expanded_local_topk = local_topk.repeat_interleave(decode_query_len, dim=0)

        reference_out = torch.empty(
            (num_tokens, 16, 128), dtype=torch.bfloat16, device=device
        )
        reference_lse = torch.empty(
            (num_tokens, 16), dtype=torch.float32, device=device
        )
        for query_offset in range(decode_query_len):
            step_reference_out = torch.empty(
                (3, 16, 128), dtype=torch.bfloat16, device=device
            )
            step_reference_lse = torch.empty(
                (3, 16), dtype=torch.float32, device=device
            )
            sparse_attention_msa_module.msa_flashinfer_sparse_decode(
                global_query[query_offset::decode_query_len].contiguous(),
                full_cache,
                expanded_global_topk[:, query_offset::decode_query_len].contiguous(),
                full_table,
                global_seq_lens,
                1,
                scale=128**-0.5,
                out=step_reference_out,
                lse_out=step_reference_lse,
            )
            reference_out[query_offset::decode_query_len].copy_(step_reference_out)
            reference_lse[query_offset::decode_query_len].copy_(step_reference_lse)

        current_key = current_value = None
        query_start_loc = None
        context_seq_lens = None
        if decode_query_len > 1:
            from vllm.v1.attention.backends.fa_utils import (
                flash_attn_varlen_func,
                get_flash_attn_version,
            )
            from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

            current_key = torch.randn(
                (num_tokens, 1, 128), generator=generator, dtype=torch.float32
            ).to(device=device, dtype=torch.bfloat16)
            current_value = torch.randn(
                (num_tokens, 1, 128), generator=generator, dtype=torch.float32
            ).to(device=device, dtype=torch.bfloat16)
            query_start_loc = torch.arange(
                0,
                num_tokens + 1,
                decode_query_len,
                dtype=torch.int32,
                device=device,
            )
            suffix_out = torch.empty_like(reference_out)
            suffix_out, suffix_lse = flash_attn_varlen_func(
                q=global_query.to(torch.bfloat16),
                k=current_key,
                v=current_value,
                out=suffix_out,
                cu_seqlens_q=query_start_loc,
                cu_seqlens_k=query_start_loc,
                max_seqlen_q=decode_query_len,
                max_seqlen_k=decode_query_len,
                causal=True,
                softmax_scale=128**-0.5,
                return_softmax_lse=True,
                fa_version=get_flash_attn_version(head_size=128),
                num_splits=1,
            )
            merged_reference = torch.empty_like(reference_out)
            merge_attn_states(
                merged_reference,
                reference_out,
                reference_lse.transpose(0, 1).contiguous(),
                suffix_out,
                suffix_lse,
            )
            reference_out = merged_reference
            context_seq_lens = local_seq_lens

        # Full CUDA graphs can append zero-query padding requests. Keep one
        # trailing padded row in every DQL case and verify that only active
        # request rows are sent to FlashInfer and dense suffix attention.
        padded_local_seq_lens = torch.cat(
            (local_seq_lens, torch.zeros(1, dtype=torch.int32, device=device))
        )
        padded_local_table = torch.cat(
            (local_table, torch.zeros_like(local_table[:1])), dim=0
        )
        padded_query_start_loc = None
        if decode_query_len > 1:
            assert query_start_loc is not None
            padded_query_start_loc = torch.cat((query_start_loc, query_start_loc[-1:]))

        metadata = MiniMaxM3SparseMetadata(
            seq_lens=padded_local_seq_lens,
            max_seq_len=3072,
            slot_mapping=torch.arange(num_tokens, device=device),
            num_actual_tokens=num_tokens,
            num_decodes=4,
            num_decode_tokens=num_tokens,
            num_prefills=0,
            num_prefill_tokens=0,
            decode=MiniMaxM3SparseMSADecodeMetadata(
                seq_lens=padded_local_seq_lens,
                block_table=padded_local_table,
                decode_query_len=decode_query_len,
                query_start_loc=padded_query_start_loc,
                context_seq_lens=(
                    padded_local_seq_lens if context_seq_lens is not None else None
                ),
            ),
        )
        layer = SimpleNamespace(
            layer_name="model.layers.0.self_attn",
            topk_indices_buffer=expanded_local_topk,
            _q_scale_float=1.0,
            _k_scale_float=1.0,
            _v_scale_float=1.0,
        )
        observed: dict[str, Any] = {}
        real_combine = sparse_attention_msa_module.cp_lse_ag_out_rs

        def record_combine(partial_out, partial_lse, cp_group, **kwargs):
            observed["partial_out"] = partial_out.clone()
            observed["partial_lse"] = partial_lse.clone()
            return real_combine(partial_out, partial_lse, cp_group, **kwargs)

        monkeypatch.setattr(
            sparse_attention_msa_module,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata={layer.layer_name: metadata}),
        )
        monkeypatch.setattr(sparse_attention_msa_module, "get_dcp_group", lambda: group)
        monkeypatch.setattr(
            sparse_attention_msa_module, "cp_lse_ag_out_rs", record_combine
        )

        output = torch.empty((num_tokens, 8 * 128), dtype=torch.bfloat16, device=device)
        _dcp_impl(rank, device, num_tokens).forward(
            layer,
            local_query.to(torch.bfloat16).view(num_tokens, -1),
            local_cache,
            output,
            query_fp8=local_query.view(num_tokens, -1),
            current_key=current_key,
            current_value=current_value,
        )

        expected = reference_out[:, rank * 8 : (rank + 1) * 8]
        torch.testing.assert_close(
            output.view(num_tokens, 8, 128), expected, rtol=2e-2, atol=2e-2
        )
        assert observed["partial_out"].shape == (num_tokens, 16, 128)
        assert observed["partial_lse"].shape == (num_tokens, 16)
        if rank == 1:
            torch.testing.assert_close(
                observed["partial_out"][decode_query_len : 2 * decode_query_len],
                torch.zeros_like(
                    observed["partial_out"][decode_query_len : 2 * decode_query_len]
                ),
            )
            assert torch.isneginf(
                observed["partial_lse"][decode_query_len : 2 * decode_query_len]
            ).all()
        assert group.all_gather_calls == 2
        assert group.reduce_scatter_calls == 1
        assert group.all_reduce_calls == 0
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not current_platform.is_cuda() or int(os.getenv("WORLD_SIZE", "1")) != 2,
    reason="requires torchrun with two CUDA ranks",
)
@pytest.mark.parametrize("num_reqs", [1, 56])
def test_minimax_m3_flashinfer_dcp2_dql4_cudagraph_replay(monkeypatch, num_reqs):
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(local_rank)
    device = torch.device("cuda", local_rank)
    init_distributed_environment(
        world_size=2,
        rank=int(os.environ["RANK"]),
        local_rank=local_rank,
        backend="nccl",
    )
    with set_current_vllm_config(VllmConfig()):
        initialize_model_parallel(
            tensor_model_parallel_size=2,
            decode_context_model_parallel_size=2,
        )
    rank = dist.get_rank()
    group = get_dcp_group()

    try:
        decode_query_len = 4
        num_tokens = num_reqs * decode_query_len
        pages_per_rank = 8
        generator = torch.Generator(device="cpu").manual_seed(101)
        local_cache = torch.randn(
            (num_reqs * pages_per_rank, 1, 128, 256),
            generator=generator,
            dtype=torch.float32,
        ).mul_(0.25)
        local_cache = local_cache.to(torch.float8_e4m3fn).to(device)
        local_table = torch.arange(
            num_reqs * pages_per_rank, dtype=torch.int32, device=device
        ).view(num_reqs, pages_per_rank)
        local_topk = torch.full(
            (num_tokens, 1, 16), -1, dtype=torch.int32, device=device
        )
        local_topk[:, 0, :pages_per_rank] = torch.arange(
            pages_per_rank, dtype=torch.int32, device=device
        )
        local_seq_lens = torch.full((num_reqs,), 1024, dtype=torch.int32, device=device)
        query_start_loc = torch.arange(
            0,
            num_tokens + 1,
            decode_query_len,
            dtype=torch.int32,
            device=device,
        )
        padded_local_table = torch.cat(
            (local_table, torch.zeros_like(local_table[:1])), dim=0
        )
        padded_local_seq_lens = torch.cat(
            (local_seq_lens, torch.zeros(1, dtype=torch.int32, device=device))
        )
        padded_query_start_loc = torch.cat((query_start_loc, query_start_loc[-1:]))
        global_query = torch.randn(
            (num_tokens, 16, 128), generator=generator, dtype=torch.float32
        ).mul_(0.25)
        global_query = global_query.to(torch.float8_e4m3fn).to(device)
        local_query = global_query[:, rank * 8 : (rank + 1) * 8].contiguous()
        local_query_bf16 = local_query.to(torch.bfloat16).view(num_tokens, -1)
        current_key = torch.randn(
            (num_tokens, 1, 128), generator=generator, dtype=torch.float32
        ).to(device=device, dtype=torch.bfloat16)
        current_value = torch.randn(
            (num_tokens, 1, 128), generator=generator, dtype=torch.float32
        ).to(device=device, dtype=torch.bfloat16)

        metadata = MiniMaxM3SparseMetadata(
            seq_lens=padded_local_seq_lens,
            max_seq_len=1024,
            slot_mapping=torch.arange(num_tokens, device=device),
            num_actual_tokens=num_tokens,
            num_decodes=num_reqs + 1,
            num_decode_tokens=num_tokens,
            num_prefills=0,
            num_prefill_tokens=0,
            decode=MiniMaxM3SparseMSADecodeMetadata(
                seq_lens=padded_local_seq_lens,
                block_table=padded_local_table,
                decode_query_len=decode_query_len,
                query_start_loc=padded_query_start_loc,
                context_seq_lens=padded_local_seq_lens,
            ),
        )
        layer = SimpleNamespace(
            layer_name="model.layers.0.self_attn",
            topk_indices_buffer=local_topk,
            _q_scale_float=1.0,
            _k_scale_float=1.0,
            _v_scale_float=1.0,
        )
        monkeypatch.setattr(
            sparse_attention_msa_module,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata={layer.layer_name: metadata}),
        )
        monkeypatch.setattr(sparse_attention_msa_module, "get_dcp_group", lambda: group)

        init_workspace_manager(device)
        impl = _dcp_impl(rank, device, num_tokens)
        impl.workspace = MSASparseAttentionWorkspace(
            56 * decode_query_len,
            16,
            128,
            torch.bfloat16,
        )
        workspace_ptrs = tuple(
            tensor.data_ptr() for tensor in impl.workspace.get(num_tokens)
        )
        output = torch.empty((num_tokens, 8 * 128), dtype=torch.bfloat16, device=device)

        def run_forward() -> None:
            impl.forward(
                layer,
                local_query_bf16,
                local_cache,
                output,
                query_fp8=local_query.view(num_tokens, -1),
                current_key=current_key,
                current_value=current_value,
            )

        run_forward()
        torch.accelerator.synchronize()
        expected = output.clone()
        dist.barrier()

        with graph_capture(device=device) as graph_capture_context:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=graph_capture_context.stream):
                run_forward()
        torch.accelerator.synchronize()
        assert workspace_ptrs == tuple(
            tensor.data_ptr() for tensor in impl.workspace.get(num_tokens)
        )

        graph.replay()
        torch.accelerator.synchronize()
        allocated_after_first_replay = torch.accelerator.memory_allocated(device)
        for _ in range(3):
            output.fill_(float("nan"))
            graph.replay()
            torch.accelerator.synchronize()
            torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)
        assert (
            torch.accelerator.memory_allocated(device) == allocated_after_first_replay
        )
    finally:
        cleanup_dist_env_and_memory()
