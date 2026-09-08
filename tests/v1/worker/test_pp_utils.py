# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which rows the PP sampled-token broadcast must carry."""

from collections import deque
from contextlib import nullcontext
from unittest.mock import MagicMock, Mock, call

import numpy as np
import torch

from vllm.v1.worker.gpu import pp_utils


def _batch(num_computed, prefill_len, num_scheduled):
    return Mock(
        num_reqs=len(num_computed),
        num_computed_tokens_np=np.array(num_computed, dtype=np.int32),
        prefill_len_np=np.array(prefill_len, dtype=np.int32),
        num_scheduled_tokens=np.array(num_scheduled, dtype=np.int32),
    )


def test_excludes_non_final_prefill_chunks():
    """Unchanged behaviour: a chunk that does not finish its prefill is skipped."""
    # Row 0 is a middle prefill chunk and produces no sample; row 1 finishes its
    # prefill this step and therefore does.
    batch = _batch(
        num_computed=[512, 1000],
        prefill_len=[4096, 1004],
        num_scheduled=[448, 4],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [False, True]


def test_none_when_no_row_samples():
    """Unchanged behaviour: an all-prefill batch needs no broadcast at all."""
    batch = _batch(
        num_computed=[0, 512],
        prefill_len=[4096, 4096],
        num_scheduled=[448, 448],
    )

    assert pp_utils.compute_need_sampled_mask(batch) is None


def test_keeps_decoding_request_past_its_length_cap():
    """A decoding request must never be dropped from the broadcast.

    Speculative decoding advances `num_computed_tokens` several tokens per step,
    so it can overrun `prompt_len + max_tokens` while the scheduler is still
    running the request. Predicting "this one is finishing" and skipping its
    broadcast freezes the earlier pipeline stages' `last_sampled_tokens` and
    `draft_tokens` while the last rank keeps advancing its own, and the stages
    then diverge permanently.
    """
    batch = _batch(
        # 14176 computed tokens is already past this request's own
        # prompt_len + max_tokens; the scheduler is still running it.
        num_computed=[14176],
        prefill_len=[12175],
        num_scheduled=[8],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True]


def test_decode_row_ahead_of_a_prefill_chunk():
    """Row order does not matter: only whether the row finishes its prefill."""
    batch = _batch(
        num_computed=[10, 512],
        prefill_len=[8, 4096],
        num_scheduled=[1, 448],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True, False]


def test_broadcast_uses_persistent_buffers(monkeypatch):
    """Side-stream NCCL sends must use storage owned by the handler."""
    handler = object.__new__(pp_utils.PPHandler)
    handler.is_last_rank = True
    handler.last_rank = 1
    handler.max_sample_len = 8
    handler.main_stream = MagicMock()
    handler.broadcast_stream = MagicMock()
    handler.broadcast_group = MagicMock()
    handler.send_sampled_tokens = MagicMock()
    handler.send_metadata = MagicMock()

    sampled_token_ids = MagicMock()
    sampled_token_ids.dtype = torch.int64
    sampled_token_ids.shape = (2, 4)
    num_sampled = MagicMock()
    num_rejected = MagicMock()
    batch = _batch(
        num_computed=[8, 9],
        prefill_len=[8, 8],
        num_scheduled=[1, 1],
    )

    monkeypatch.setattr(pp_utils.current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    broadcast = MagicMock()
    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)

    handler.broadcast(sampled_token_ids, num_sampled, num_rejected, batch)

    send_tokens = handler.send_sampled_tokens.__getitem__.return_value
    send_metadata = handler.send_metadata.__getitem__.return_value.view.return_value
    broadcast.assert_has_calls(
        [
            call(send_tokens, src=1, group=handler.broadcast_group),
            call(send_metadata, src=1, group=handler.broadcast_group),
        ]
    )
    send_tokens.zero_.assert_called_once_with()
    send_tokens.__getitem__.return_value.copy_.assert_called_once_with(
        sampled_token_ids
    )
    send_metadata.__getitem__.return_value.copy_.assert_has_calls(
        [call(num_sampled), call(num_rejected)]
    )


def test_filtered_receive_waits_before_early_return():
    """Receive storage cannot be released before its side-stream work completes."""
    event = MagicMock()
    slot = pp_utils.PendingRecv(
        event=event,
        sampled_tokens=MagicMock(),
        num_sampled=MagicMock(),
        num_rejected=MagicMock(),
        idx_mapping=MagicMock(),
        idx_mapping_np=np.array([0], dtype=np.int32),
        need_sampled_mask=np.array([False]),
        gen_at_receive_np=np.array([0], dtype=np.int32),
    )
    handler = object.__new__(pp_utils.PPHandler)
    handler.queue = deque([slot])
    handler.main_stream = MagicMock()
    handler.req_idx_gen_np = np.array([0], dtype=np.int32)

    assert handler.get_prev_sampled_outputs() is None
    handler.main_stream.wait_event.assert_called_once_with(event)
