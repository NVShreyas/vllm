# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.models.minimax_m3.nvidia.model import MiniMaxM3SparseAttention
from vllm.platforms import current_platform
from vllm.tracing import instrument

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


@instrument(span_name="MiniMax M3 MSA warmup")
def minimax_m3_msa_warmup(worker: "Worker") -> None:
    sparse_module = next(
        (
            module
            for module in worker.get_model().modules()
            if isinstance(module, MiniMaxM3SparseAttention)
        ),
        None,
    )
    if sparse_module is None:
        return
    if not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(100)
    ):
        return

    logger.info("Warming up MiniMax M3 MSA kernels.")

    # Cover sparse prefill through the normal model path.
    worker.model_runner._dummy_run(
        num_tokens=16,
        skip_eplb=True,
        is_profile=True,
        force_attention=True,
        create_mixed_batch=True,
    )

    # Compile the decode-only kernels and reserve the largest workspace used by
    # the supported concurrency range before the global workspace is locked and
    # CUDA graph capture starts. The configured DQL covers both ordinary decode
    # (1) and Eagle verification (up to 4) through the normal runner metadata.
    runner = worker.model_runner
    decode_query_len = getattr(
        runner,
        "decode_query_len",
        getattr(runner, "uniform_decode_query_len", 1),
    )
    max_tokens = min(
        worker.scheduler_config.max_num_batched_tokens,
        56 * decode_query_len,
    )
    for num_tokens in sorted({decode_query_len, max_tokens}):
        runner._dummy_run(
            num_tokens=num_tokens,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            force_attention=True,
            uniform_decode=True,
            skip_eplb=True,
        )
