# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility methods for model layers."""
from typing import Callable, Optional

import torch

from vllm import _custom_ops as ops
from vllm import envs
from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op


def get_token_bin_counts_and_mask(
    tokens: torch.Tensor,
    vocab_size: int,
    num_seqs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Compute the bin counts for the tokens.
    # vocab_size + 1 for padding.
    bin_counts = torch.zeros((num_seqs, vocab_size + 1),
                             dtype=torch.long,
                             device=tokens.device)
    bin_counts.scatter_add_(1, tokens, torch.ones_like(tokens))
    bin_counts = bin_counts[:, :vocab_size]
    mask = bin_counts > 0

    return bin_counts, mask


def apply_penalties(logits: torch.Tensor, prompt_tokens_tensor: torch.Tensor,
                    output_tokens_tensor: torch.Tensor,
                    presence_penalties: torch.Tensor,
                    frequency_penalties: torch.Tensor,
                    repetition_penalties: torch.Tensor) -> torch.Tensor:
    """
    Applies penalties in place to the logits tensor
    logits : The input logits tensor of shape [num_seqs, vocab_size]
    prompt_tokens_tensor: A tensor containing the prompt tokens. The prompts 
        are padded to the maximum prompt length within the batch using 
        `vocab_size` as the padding value. The value `vocab_size` is used 
        for padding because it does not correspond to any valid token ID 
        in the vocabulary.
    output_tokens_tensor: The output tokens tensor.
    presence_penalties: The presence penalties of shape (num_seqs, )
    frequency_penalties: The frequency penalties of shape (num_seqs, )
    repetition_penalties: The repetition penalties of shape (num_seqs, )
    """
    num_seqs, vocab_size = logits.shape
    _, prompt_mask = get_token_bin_counts_and_mask(prompt_tokens_tensor,
                                                   vocab_size, num_seqs)
    output_bin_counts, output_mask = get_token_bin_counts_and_mask(
        output_tokens_tensor, vocab_size, num_seqs)

    # Apply repetition penalties as a custom op
    from vllm._custom_ops import apply_repetition_penalties
    apply_repetition_penalties(logits, prompt_mask, output_mask,
                               repetition_penalties)

    # We follow the definition in OpenAI API.
    # Refer to https://platform.openai.com/docs/api-reference/parameter-details
    logits -= frequency_penalties.unsqueeze(dim=1) * output_bin_counts
    logits -= presence_penalties.unsqueeze(dim=1) * output_mask
    return logits


def default_unquantized_gemm(layer: torch.nn.Module,
                             x: torch.Tensor,
                             weight: torch.Tensor,
                             bias: Optional[torch.Tensor] = None):
    return torch.nn.functional.linear(x, weight, bias)


def rocm_unquantized_gemm_impl(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    from vllm.platforms.rocm import on_gfx9
    k = weight.shape[1]
    use_skinny = (envs.VLLM_ROCM_USE_SKINNY_GEMM and on_gfx9() and \
                    x.dtype in [torch.float16, torch.bfloat16] \
                    and k % 8 == 0 and bias is None)

    if use_skinny is not True:
        return torch.nn.functional.linear(x, weight, bias)

    x_view = x.view(-1, x.size(-1))
    n = x_view.shape[0]
    m = weight.shape[0]
    cu_count = current_platform.get_cu_count()

    if m > 8 and 0 < n <= 4:
        out = ops.wvSplitK(weight, x_view, cu_count)
        return out.view(*x.shape[:-1], weight.shape[0])
    elif m % 4 == 0 and n == 1 and k <= 8192:
        out = ops.LLMM1(weight, x_view, 4)
        return out.view(*x.shape[:-1], weight.shape[0])
    return torch.nn.functional.linear(x, weight, bias)


def rocm_unquantized_gemm_impl_fake(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def rocm_unquantized_gemm(layer: torch.nn.Module,
                          x: torch.Tensor,
                          weight: torch.Tensor,
                          bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.ops.vllm.rocm_unquantized_gemm_impl(x, weight, bias)


direct_register_custom_op(
    op_name="rocm_unquantized_gemm_impl",
    op_func=rocm_unquantized_gemm_impl,
    mutates_args=[],
    fake_impl=rocm_unquantized_gemm_impl_fake,
    dispatch_key=current_platform.dispatch_key,
)


def cpu_unquantized_gemm(layer: torch.nn.Module,
                         x: torch.Tensor,
                         weight: torch.Tensor,
                         bias: Optional[torch.Tensor] = None):
    if getattr(layer, "use_cpu_sgl", False):
        return torch.ops._C.weight_packed_linear(x, weight, bias, True)
    else:
        return torch.nn.functional.linear(x, weight, bias)


def dispatch_unquantized_gemm() -> Callable[..., torch.Tensor]:
    if current_platform.is_rocm():
        return rocm_unquantized_gemm
    elif current_platform.is_cpu():
        return cpu_unquantized_gemm
    else:
        return default_unquantized_gemm
