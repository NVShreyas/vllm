# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MSA (SM100/Blackwell) block-sparse attention for MiniMax M3.

Prefill attends with ``fmha_sm100`` (``build_k2q_csr`` + ``sparse_atten_func``).
Decode uses Triton split-K by default, with opt-in CUTLASS ``fmha_sm100`` or
FlashInfer MSA paths for regular decode and speculative verification.
"""

from dataclasses import dataclass

import torch

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.attention import MiniMaxM3MSADecodeBackend
from vllm.distributed import get_dcp_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.models.minimax_m3.common.dcp import (
    minimax_m3_dcp_prefill_segments,
    minimax_m3_decode_seq_lens,
)
from vllm.models.minimax_m3.common.ops.sparse_attn import (
    SPARSE_BLOCK_SIZE,
    minimax_m3_sparse_attn_decode,
)
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparseBackend,
    MiniMaxM3SparseDecodeMetadata,
    MiniMaxM3SparseImpl,
    MiniMaxM3SparseMetadata,
    MiniMaxM3SparseMetadataBuilder,
    MiniMaxM3SparsePrefillMetadata,
)
from vllm.models.minimax_m3.nvidia.msa_cutlass_sparse_decode import (
    MSACutlassDecodeMetadata,
    MSACutlassDecodePlanCache,
    msa_cutlass_sparse_decode,
    prepare_decode_metadata,
    should_prepare_decode_metadata,
    supports_cutlass_sparse_decode,
)
from vllm.models.minimax_m3.nvidia.msa_flashinfer_sparse_decode import (
    msa_flashinfer_sparse_decode,
    supports_flashinfer_sparse_decode,
)
from vllm.models.minimax_m3.nvidia.ops import write_dcp_local_seq_lens
from vllm.v1.attention.backend import (
    AttentionLayer,
    CommonAttentionMetadata,
)
from vllm.v1.attention.ops.dcp import (
    cp_lse_ag_out_rs,
    get_dcp_workspace_max_num_tokens,
    mask_dcp_empty_shards_,
)
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)


class MiniMaxM3SparseMSABackend(MiniMaxM3SparseBackend):
    """MiniMax M3 backend with NVIDIA MSA-specific decode metadata."""

    @staticmethod
    def get_builder_cls() -> type["MiniMaxM3SparseMSAMetadataBuilder"]:
        return MiniMaxM3SparseMSAMetadataBuilder


class MiniMaxM3SparseCutlassBackend(MiniMaxM3SparseMSABackend):
    """Attention-backend alias selecting CUTLASS MSA sparse decode."""

    @staticmethod
    def get_name() -> str:
        return "CUTLASS_MSA"


class MiniMaxM3SparseFlashInferBackend(MiniMaxM3SparseMSABackend):
    """Attention-backend alias selecting FlashInfer MSA sparse decode."""

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MSA"


class MiniMaxM3SparseTritonBackend(MiniMaxM3SparseMSABackend):
    """Attention-backend alias selecting Triton MSA sparse decode."""

    @staticmethod
    def get_name() -> str:
        return "TRITON_MSA"


@dataclass
class MiniMaxM3SparseMSADecodeMetadata(MiniMaxM3SparseDecodeMetadata):
    msa_cutlass: MSACutlassDecodeMetadata | None = None
    query_start_loc: torch.Tensor | None = None
    context_seq_lens: torch.Tensor | None = None


@dataclass
class MiniMaxM3SparseMSADCPPrefillGroup:
    """One causal mode of a page-boundary-segmented DCP prefill."""

    query_indices: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    max_query_len: int
    max_seq_len: int
    total_kv_blocks: int
    causal: bool


@dataclass
class MiniMaxM3SparseMSAPrefillMetadata(MiniMaxM3SparsePrefillMetadata):
    dcp_groups: tuple[MiniMaxM3SparseMSADCPPrefillGroup, ...] = ()


class MiniMaxM3SparseMSAMetadataBuilder(MiniMaxM3SparseMetadataBuilder):
    """Prepare MSA plans only for decode shapes supported by ``fmha_sm100``."""

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        config = vllm_config.model_config.hf_text_config
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.num_q_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = kv_cache_spec.num_kv_heads
        self.topk_blocks = config.sparse_attention_config["sparse_topk_blocks"]
        # AttentionSpec stores every FP8 mode as uint8, so retain the configured
        # format to distinguish E4M3 (supported) from E5M2 before planning.
        self.kv_cache_dtype = vllm_config.cache_config.cache_dtype
        self.decode_backend = vllm_config.attention_config.minimax_m3_msa_decode_backend
        self.msa_cutlass_plan_cache = MSACutlassDecodePlanCache()
        parallel_config = vllm_config.parallel_config
        self.dcp_world_size = parallel_config.decode_context_parallel_size
        self.cp_kv_cache_interleave_size = parallel_config.cp_kv_cache_interleave_size
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        self.dcp_context_seq_lens_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_seqs,
            dtype=torch.int32,
            device=device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MiniMaxM3SparseMetadata:
        metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            fast_build,
        )
        prefill = metadata.prefill
        if prefill is not None and self.dcp_world_size > 1:
            seq_lens_cpu = (
                common_attn_metadata.seq_lens[
                    : metadata.num_decodes + metadata.num_prefills
                ]
                .cpu()
                .to(torch.int32)
            )
            segments = minimax_m3_dcp_prefill_segments(
                common_attn_metadata.query_start_loc_cpu,
                seq_lens_cpu,
                num_decodes=metadata.num_decodes,
                dcp_rank=self.dcp_rank,
                dcp_world_size=self.dcp_world_size,
                interleave_size=self.cp_kv_cache_interleave_size,
            )
            device = prefill.seq_lens.device
            groups: list[MiniMaxM3SparseMSADCPPrefillGroup] = []
            for causal, segment_mask in (
                (True, segments.owned),
                (False, ~segments.owned),
            ):
                segment_ids = torch.nonzero(segment_mask, as_tuple=False).flatten()
                if segment_ids.numel() == 0:
                    continue
                query_lens = segments.query_lens[segment_ids]
                kv_lens_cpu = segments.kv_lens[segment_ids]
                query_indices_cpu = torch.cat(
                    [
                        torch.arange(start, start + length, dtype=torch.int64)
                        for start, length in zip(
                            segments.query_starts[segment_ids].tolist(),
                            query_lens.tolist(),
                        )
                    ]
                )
                request_indices = segments.request_indices[segment_ids].to(device)
                group_block_table = prefill.block_table.index_select(0, request_indices)
                kv_lens = kv_lens_cpu.to(device)
                cu_seqlens_q = torch.cat(
                    (
                        torch.zeros(1, dtype=torch.int32, device=device),
                        query_lens.to(device).cumsum(0, dtype=torch.int32),
                    )
                )
                cu_seqlens_k = torch.cat(
                    (
                        torch.zeros(1, dtype=torch.int32, device=device),
                        kv_lens.cumsum(0, dtype=torch.int32),
                    )
                )
                groups.append(
                    MiniMaxM3SparseMSADCPPrefillGroup(
                        query_indices=query_indices_cpu.to(device),
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        seq_lens=kv_lens,
                        block_table=group_block_table,
                        max_query_len=int(query_lens.max()),
                        max_seq_len=int(kv_lens_cpu.max()),
                        total_kv_blocks=int(
                            ((kv_lens_cpu + SPARSE_BLOCK_SIZE - 1) // SPARSE_BLOCK_SIZE)
                            .sum()
                            .item()
                        ),
                        causal=causal,
                    )
                )
            metadata.prefill = MiniMaxM3SparseMSAPrefillMetadata(
                **prefill.__dict__, dcp_groups=tuple(groups)
            )
        decode = metadata.decode
        if decode is None:
            return metadata

        # Only FlashInfer can provide the partial output and LSE required for
        # DCP reduction. CUTLASS and Triton retain their global metadata and are
        # rejected by the generic DCP compatibility check.
        if (
            self.decode_backend == "flashinfer"
            and common_attn_metadata.dcp_local_seq_lens is not None
        ):
            decode.seq_lens = minimax_m3_decode_seq_lens(
                common_attn_metadata, metadata.num_decodes
            )

        context_seq_lens = None
        query_start_loc = None
        if self.decode_backend == "flashinfer" and self.dcp_world_size > 1:
            query_start_loc = common_attn_metadata.query_start_loc[
                : metadata.num_decodes + 1
            ]
            context_seq_lens = self.dcp_context_seq_lens_buffer[: metadata.num_decodes]
            write_dcp_local_seq_lens(
                common_attn_metadata.seq_lens[: metadata.num_decodes],
                context_seq_lens,
                global_offset=-decode.decode_query_len,
                dcp_rank=self.dcp_rank,
                dcp_world_size=self.dcp_world_size,
                interleave_size=self.cp_kv_cache_interleave_size,
            )

        msa_cutlass = None
        if should_prepare_decode_metadata(
            metadata.num_decodes,
            decode.decode_query_len,
            decode_backend=self.decode_backend,
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            kv_cache_dtype=self.kv_cache_dtype,
            page_size=SPARSE_BLOCK_SIZE,
            topk_blocks=self.topk_blocks,
        ):
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            assert seq_lens_cpu is not None
            msa_cutlass = prepare_decode_metadata(
                decode.block_table,
                decode.seq_lens,
                seq_lens_cpu[: metadata.num_decodes],
                decode.decode_query_len,
                num_q_heads=self.num_q_heads,
                num_kv_heads=self.num_kv_heads,
                page_size=SPARSE_BLOCK_SIZE,
                topk_blocks=self.topk_blocks,
                plan_cache=self.msa_cutlass_plan_cache,
            )
        metadata.decode = MiniMaxM3SparseMSADecodeMetadata(
            seq_lens=decode.seq_lens,
            block_table=decode.block_table,
            decode_query_len=decode.decode_query_len,
            msa_cutlass=msa_cutlass,
            query_start_loc=query_start_loc,
            context_seq_lens=context_seq_lens,
        )
        return metadata


class MSASparseAttentionWorkspace:
    """Stable scratch storage for graph-captured MSA sparse attention."""

    def __init__(
        self,
        max_num_tokens: int,
        context_num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> None:
        self.max_num_tokens = max_num_tokens
        self.context_num_heads = context_num_heads
        self.head_size = head_size
        self.dtype = dtype

    def get(self, num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        if num_tokens > self.max_num_tokens:
            raise ValueError(
                "MiniMax M3 MSA sparse attention workspace capacity exceeded: "
                f"{num_tokens} > {self.max_num_tokens}"
            )
        context_out, context_lse = current_workspace_manager().get_simultaneous(
            (
                (
                    self.max_num_tokens,
                    self.context_num_heads,
                    self.head_size,
                ),
                self.dtype,
            ),
            (
                (self.max_num_tokens, self.context_num_heads),
                torch.float32,
            ),
        )
        return context_out[:num_tokens], context_lse[:num_tokens]


class MiniMaxM3SparseMSAImpl(MiniMaxM3SparseImpl):
    """MSA block-sparse attention with guarded CUTLASS sparse decode."""

    supports_dcp = True
    can_return_lse_for_decode = True
    lse_base_on_e = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        kv_cache_dtype: str = "auto",
        *,
        topk_blocks: int,
        sparse_block_size: int,
        msa_decode_backend: MiniMaxM3MSADecodeBackend = "triton",
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            kv_cache_dtype,
            topk_blocks=topk_blocks,
            sparse_block_size=sparse_block_size,
        )
        self.use_cutlass_decode = supports_cutlass_sparse_decode(
            decode_backend=msa_decode_backend,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            kv_cache_dtype=self.kv_cache_dtype,
            page_size=self.block_size,
            topk_blocks=self.topk_blocks,
        )
        self.use_flashinfer_decode = supports_flashinfer_sparse_decode(
            decode_backend=msa_decode_backend,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            kv_cache_dtype=self.kv_cache_dtype,
            page_size=self.block_size,
            topk_blocks=self.topk_blocks,
        )
        # Only FlashInfer exposes the partial output and softmax LSE needed to
        # combine rank-local KV shards. Keep CUTLASS and Triton rejected by the
        # generic DCP compatibility check even though this concrete MSA class
        # advertises the FlashInfer capability.
        self.need_to_return_lse_for_decode = (
            self.dcp_world_size > 1 and self.use_flashinfer_decode
        )
        if self.need_to_return_lse_for_decode:
            context_num_heads = self.num_heads * self.dcp_world_size
            if context_num_heads % self.num_kv_heads != 0:
                raise ValueError(
                    "MiniMax M3 FlashInfer DCP requires the gathered query-head "
                    "count to be divisible by the rank-local KV-head count, got "
                    f"{context_num_heads} and {self.num_kv_heads}."
                )
            if context_num_heads // self.num_kv_heads > 16:
                raise ValueError(
                    "MiniMax M3 FlashInfer DCP supports at most 16 gathered "
                    "query heads per rank-local KV head, got "
                    f"{context_num_heads // self.num_kv_heads}."
                )
            vllm_config = get_current_vllm_config()
            self.workspace = MSASparseAttentionWorkspace(
                # Decode needs at most the generic DCP graph-workspace capacity,
                # while sparse prefill can contain the full token budget. The
                # same stable buffers serve both paths.
                max(
                    get_dcp_workspace_max_num_tokens(vllm_config),
                    vllm_config.scheduler_config.max_num_batched_tokens,
                ),
                context_num_heads,
                self.head_size,
                vllm_config.model_config.dtype,
            )
        selected_backend = (
            "FlashInfer"
            if self.use_flashinfer_decode
            else "CUTLASS"
            if self.use_cutlass_decode
            else "Triton"
        )
        logger.info_once(
            "MiniMax M3 MSA sparse decode selected %s",
            selected_backend,
        )

    def should_use_msa_decode(self, layer_name: str) -> bool:
        use_cutlass_decode = getattr(self, "use_cutlass_decode", False)
        use_flashinfer_decode = getattr(self, "use_flashinfer_decode", False)
        if not (use_cutlass_decode or use_flashinfer_decode):
            return False
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return False
        main_md = attn_metadata[layer_name]
        if not isinstance(main_md, MiniMaxM3SparseMetadata):
            return False
        decode = main_md.decode
        if use_flashinfer_decode:
            return isinstance(decode, MiniMaxM3SparseMSADecodeMetadata)
        return (
            isinstance(decode, MiniMaxM3SparseMSADecodeMetadata)
            and decode.msa_cutlass is not None
        )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        *,
        query_fp8: torch.Tensor | None = None,
        current_key: torch.Tensor | None = None,
        current_value: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return output  # profiling run; caches unbound
        main_md = attn_metadata[layer.layer_name]  # type: ignore[attr-defined]
        assert isinstance(main_md, MiniMaxM3SparseMetadata)

        nd = main_md.num_decode_tokens
        num_tokens = main_md.num_actual_tokens
        # Indexer top-k from the shared token-major buffer [total_q, H, MK]; the
        # kernels want [H, tokens, MK], so slice tokens on dim 0 then transpose.
        topk = layer.topk_indices_buffer  # type: ignore[attr-defined]
        assert topk is not None
        hd = self.head_size
        q = query[:num_tokens].view(-1, self.num_heads, hd)
        out = output[:num_tokens].view(-1, self.num_heads, hd)
        kv_cache = (
            kv_cache.view(self.kv_cache_fp8_dtype) if self.use_fp8_kv else kv_cache
        )
        k_scale = getattr(layer, "_k_scale", None) if self.use_fp8_kv else None
        v_scale = getattr(layer, "_v_scale", None) if self.use_fp8_kv else None

        # Decode [:nd]: CUTLASS for planned shapes, otherwise Triton.
        if main_md.num_decodes > 0:
            d = main_md.decode
            assert d is not None
            msa_metadata = (
                d.msa_cutlass
                if isinstance(d, MiniMaxM3SparseMSADecodeMetadata)
                else None
            )
            if getattr(self, "use_flashinfer_decode", False):
                assert isinstance(d, MiniMaxM3SparseMSADecodeMetadata)
                assert query_fp8 is not None
                active_num_decodes = nd // d.decode_query_len
                assert active_num_decodes * d.decode_query_len == nd
                use_dcp = self.need_to_return_lse_for_decode
                flashinfer_query = query_fp8[:nd].view(-1, self.num_heads, hd)
                flashinfer_out = out[:nd]
                if use_dcp:
                    flashinfer_query = get_dcp_group().all_gather(
                        flashinfer_query.contiguous(), dim=1
                    )
                    flashinfer_out, flashinfer_lse = self.workspace.get(nd)
                else:
                    flashinfer_lse = torch.empty(
                        flashinfer_query.shape[:-1],
                        dtype=torch.float32,
                        device=query.device,
                    )
                q_scale_float = getattr(layer, "_q_scale_float", 1.0)
                k_scale_float = getattr(layer, "_k_scale_float", 1.0)
                v_scale_float = getattr(layer, "_v_scale_float", 1.0)
                use_context_only = use_dcp and d.decode_query_len > 1
                if use_context_only:
                    assert d.context_seq_lens is not None

                assert isinstance(d.context_seq_lens, torch.Tensor)
                flashinfer_output, _ = msa_flashinfer_sparse_decode(
                    flashinfer_query,
                    kv_cache,
                    topk[:nd].transpose(0, 1),
                    d.block_table[:active_num_decodes],
                    (
                        d.context_seq_lens[:active_num_decodes]
                        if use_context_only
                        else d.seq_lens[:active_num_decodes]
                    ),
                    d.decode_query_len,
                    scale=self.scale,
                    q_scale_float=q_scale_float,
                    k_scale_float=k_scale_float,
                    v_scale_float=v_scale_float,
                    causal=not use_context_only,
                    out=flashinfer_out,
                    lse_out=flashinfer_lse,
                )
                assert flashinfer_output.data_ptr() == flashinfer_out.data_ptr()
                if use_dcp:
                    # A rank with no local KV tokens is the identity element of
                    # the distributed softmax. Explicitly neutralize both
                    # values before the cross-rank LSE/output combine.
                    dcp_seq_lens = (
                        d.context_seq_lens if d.decode_query_len > 1 else d.seq_lens
                    )
                    assert dcp_seq_lens is not None
                    empty_shards = dcp_seq_lens[:active_num_decodes] == 0
                    if d.decode_query_len > 1:
                        empty_shards = empty_shards.repeat_interleave(
                            d.decode_query_len
                        )
                    flashinfer_out.masked_fill_(empty_shards[:, None, None], 0)
                    flashinfer_lse.masked_fill_(empty_shards[:, None], float("-inf"))
                    combined = cp_lse_ag_out_rs(
                        flashinfer_out,
                        flashinfer_lse,
                        get_dcp_group(),
                        return_lse=d.decode_query_len > 1,
                        is_lse_base_on_e=self.lse_base_on_e,
                    )
                    if d.decode_query_len == 1:
                        assert isinstance(combined, torch.Tensor)
                        out[:nd].copy_(combined)
                    else:
                        from vllm.v1.attention.backends.fa_utils import (
                            flash_attn_varlen_func,
                            get_flash_attn_version,
                        )

                        assert isinstance(combined, tuple)
                        context_out, context_lse = combined
                        assert current_key is not None and current_value is not None
                        assert d.query_start_loc is not None
                        fa_version = get_flash_attn_version(head_size=hd)
                        assert fa_version is not None
                        suffix_out, suffix_lse = flash_attn_varlen_func(
                            q=q[:nd],
                            k=current_key[:nd].view(-1, self.num_kv_heads, hd),
                            v=current_value[:nd].view(-1, self.num_kv_heads, hd),
                            out=out[:nd],
                            cu_seqlens_q=d.query_start_loc[: active_num_decodes + 1],
                            cu_seqlens_k=d.query_start_loc[: active_num_decodes + 1],
                            max_seqlen_q=d.decode_query_len,
                            max_seqlen_k=d.decode_query_len,
                            causal=True,
                            softmax_scale=self.scale,
                            return_softmax_lse=True,
                            fa_version=fa_version,
                            num_splits=1,
                        )
                        merge_attn_states(
                            out[:nd],
                            context_out,
                            context_lse.transpose(0, 1).contiguous(),
                            suffix_out,
                            suffix_lse,
                        )
            elif self.use_cutlass_decode and msa_metadata is not None:
                assert query_fp8 is not None
                msa_cutlass_sparse_decode(
                    query_fp8[:nd].view(-1, self.num_heads, hd),
                    kv_cache,
                    topk[:nd],
                    out[:nd],
                    msa_metadata,
                    scale=self.scale,
                    q_scale_float=getattr(layer, "_q_scale_float", 1.0),
                    k_scale_float=getattr(layer, "_k_scale_float", 1.0),
                    v_scale_float=getattr(layer, "_v_scale_float", 1.0),
                )
            else:
                minimax_m3_sparse_attn_decode(
                    q[:nd],
                    kv_cache,
                    topk[:nd].transpose(0, 1),
                    d.block_table,
                    d.seq_lens,
                    self.num_kv_heads,
                    self.scale,
                    out[:nd],
                    d.decode_query_len,
                    k_scale=k_scale,
                    v_scale=v_scale,
                )

        # Prefill [nd:]: MSA sparse FMHA over the selected blocks.
        if main_md.num_prefills > 0:
            from vllm.third_party.fmha_sm100.sparse import (
                build_k2q_csr,
                sparse_atten_func,
            )

            p = main_md.prefill
            assert p is not None
            # [H, prefill, MK] transposed view; build_k2q_csr consumes the
            # strided view directly (topK stays innermost-contiguous).
            prefill_topk = topk[nd:num_tokens].transpose(0, 1)
            qp = q[nd:]
            k_cache, v_cache = kv_cache.split(self.head_size, dim=-1)
            if self.need_to_return_lse_for_decode:
                assert isinstance(p, MiniMaxM3SparseMSAPrefillMetadata)
                gathered_q = get_dcp_group().all_gather(qp.contiguous(), dim=1)
                partial_out, partial_lse = self.workspace.get(qp.shape[0])
                partial_out.zero_()
                partial_lse.fill_(float("-inf"))
                for group in p.dcp_groups:
                    group_q = gathered_q.index_select(0, group.query_indices)
                    group_topk = prefill_topk.index_select(1, group.query_indices)
                    if group.total_kv_blocks == 0:
                        continue
                    k2q_row_ptr, k2q_q_indices, schedule = build_k2q_csr(
                        group_topk,
                        group.cu_seqlens_q,
                        group.cu_seqlens_k,
                        SPARSE_BLOCK_SIZE,
                        total_k=0,
                        max_seqlen_k=group.max_seq_len,
                        max_seqlen_q=group.max_query_len,
                        total_rows=group.total_kv_blocks,
                        qhead_per_kv=group_q.shape[1] // self.num_kv_heads,
                        return_schedule=True,
                    )
                    group_out = torch.empty_like(group_q, dtype=out.dtype)
                    group_out, group_lse = sparse_atten_func(
                        group_q,
                        k_cache,
                        v_cache,
                        k2q_row_ptr,
                        k2q_q_indices,
                        topK=self.topk_blocks,
                        blk_kv=SPARSE_BLOCK_SIZE,
                        causal=group.causal,
                        softmax_scale=self.scale,
                        cu_seqlens_q=group.cu_seqlens_q,
                        cu_seqlens_k=group.cu_seqlens_k,
                        max_seqlen_q=group.max_query_len,
                        max_seqlen_k=group.max_seq_len,
                        page_table=group.block_table,
                        seqused_k=group.seq_lens,
                        schedule=schedule,
                        return_softmax_lse=True,
                        out=group_out,
                    )
                    mask_dcp_empty_shards_(
                        group_lse, group.seq_lens, group.cu_seqlens_q
                    )
                    row_indices = torch.arange(
                        group_out.shape[0],
                        device=group_out.device,
                        dtype=group.cu_seqlens_q.dtype,
                    )
                    sequence_indices = torch.searchsorted(
                        group.cu_seqlens_q[1:], row_indices, right=True
                    ).clamp_max(group.seq_lens.shape[0] - 1)
                    empty_rows = (row_indices >= group.cu_seqlens_q[-1]) | (
                        group.seq_lens[sequence_indices] == 0
                    )
                    group_out.masked_fill_(empty_rows[:, None, None], 0)
                    partial_out.index_copy_(0, group.query_indices, group_out)
                    partial_lse.index_copy_(0, group.query_indices, group_lse)
                combined = cp_lse_ag_out_rs(
                    partial_out,
                    partial_lse,
                    get_dcp_group(),
                    is_lse_base_on_e=self.lse_base_on_e,
                )
                assert isinstance(combined, torch.Tensor)
                out[nd:].copy_(combined)
            else:
                k2q_row_ptr, k2q_q_indices, schedule = build_k2q_csr(
                    prefill_topk,
                    p.cu_seqlens_q,
                    p.cu_seqlens_k,
                    SPARSE_BLOCK_SIZE,
                    total_k=0,
                    max_seqlen_k=p.max_seq_len,
                    max_seqlen_q=p.max_query_len,
                    total_rows=p.total_kv_blocks,
                    qhead_per_kv=qp.shape[1] // self.num_kv_heads,
                    return_schedule=True,
                )
                sparse_atten_func(
                    qp,
                    k_cache,
                    v_cache,
                    k2q_row_ptr,
                    k2q_q_indices,
                    topK=self.topk_blocks,
                    blk_kv=SPARSE_BLOCK_SIZE,
                    causal=True,
                    softmax_scale=self.scale,
                    cu_seqlens_q=p.cu_seqlens_q,
                    cu_seqlens_k=p.cu_seqlens_k,
                    max_seqlen_q=p.max_query_len,
                    max_seqlen_k=p.max_seq_len,
                    page_table=p.block_table,
                    seqused_k=p.seq_lens,
                    schedule=schedule,
                    out=out[nd:],
                )
        return output
