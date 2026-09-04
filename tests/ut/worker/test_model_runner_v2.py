from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_ascend.worker.v2.aclgraph_utils import ModelWithContext
from vllm_ascend.worker.v2.model_runner import (
    AscendAdaptiveVerificationManager,
    NPUModelRunner,
)


def _make_runner(need_timing: bool = True):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.ascend_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(profiling_chunk_config=SimpleNamespace(need_timing=need_timing))
    )
    runner.vllm_config = SimpleNamespace()
    runner.execute_model_state = None
    runner.is_last_pp_rank = False
    return runner


@pytest.mark.parametrize("is_vllm_0_27_1", [True, False], ids=["v0.27.1", "newer"])
def test_execute_model_records_profiling_time(is_vllm_0_27_1):
    runner = _make_runner()
    scheduler_output = SimpleNamespace(disable_profiling_timing=False)

    with (
        patch.object(
            GPUModelRunner,
            "execute_model",
            return_value=None,
        ) as mock_execute_model,
        patch(
            "vllm_ascend.worker.v2.model_runner.vllm_version_is",
            return_value=is_vllm_0_27_1,
        ),
        patch("vllm_ascend.core.profiling_chunk_predictor.torch.npu.synchronize") as mock_synchronize,
        patch(
            "vllm_ascend.core.profiling_chunk_predictor.time.perf_counter",
            side_effect=[10.0, 10.125],
        ),
    ):
        output = runner.execute_model(scheduler_output)

    assert output is None
    assert runner._cpp_execution_time_ms == pytest.approx(125.0)
    assert mock_synchronize.call_count == 2
    expected_kwargs: dict[str, object] = {
        "intermediate_tensors": None,
        "dummy_run": False,
        "skip_attn_for_dummy_run": False,
        "is_profile": False,
    }
    if not is_vllm_0_27_1:
        expected_kwargs["context_len"] = 0
    mock_execute_model.assert_called_once_with(scheduler_output, **expected_kwargs)


def test_execute_model_disables_profiling_timer_and_clears_stale_time():
    runner = _make_runner()
    runner._cpp_execution_time_ms = 123.0
    scheduler_output = SimpleNamespace(disable_profiling_timing=True)

    with (
        patch.object(
            GPUModelRunner,
            "execute_model",
            return_value=None,
        ),
        patch("vllm_ascend.core.profiling_chunk_predictor.torch.npu.synchronize") as mock_synchronize,
        patch("vllm_ascend.core.profiling_chunk_predictor.time.perf_counter") as mock_perf_counter,
    ):
        runner.execute_model(scheduler_output)

    profiling_config = runner.ascend_config.scheduler_config.profiling_chunk_config
    assert not profiling_config.need_timing
    assert runner._cpp_execution_time_ms is None
    mock_synchronize.assert_not_called()
    mock_perf_counter.assert_not_called()


def test_full_decode_only_keeps_graph_descriptor_request_count():
    runner = _make_runner()
    runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY)
    runner.decode_query_len = 1
    query_start_loc_np = np.array([0, 1, 2, 2, 2, 2], dtype=np.int32)

    actual, num_reqs_padded = runner._pad_query_start_loc_for_fia(
        num_tokens_padded=4,
        num_reqs_padded=4,
        num_reqs=2,
        query_start_loc_np=query_start_loc_np,
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        batch_desc_num_reqs=4,
    )

    assert num_reqs_padded == 4
    np.testing.assert_array_equal(actual[:5], np.array([0, 1, 2, 3, 4], dtype=np.int32))


@pytest.mark.parametrize(
    ("num_reqs", "num_reqs_padded", "expected"),
    [
        (2, 4, [0, 2, 5, 6, 8]),
        (4, 4, [0, 1, 3, 5, 8]),
    ],
)
def test_adaptive_verification_padding_ends_at_graph_token_count(
    num_reqs,
    num_reqs_padded,
    expected,
):
    runner = _make_runner()
    runner.max_num_reqs = 4
    query_start_loc_np = np.full(runner.max_num_reqs + 2, 5, dtype=np.int32)
    query_start_loc_np[:3] = [0, 2, 5]
    if num_reqs == 4:
        query_start_loc_np[:5] = [0, 1, 3, 5, 6]

    actual, num_reqs_padded = runner._pad_adaptive_query_start_loc_for_fia(
        num_tokens_padded=8,
        num_reqs_padded=num_reqs_padded,
        num_reqs=num_reqs,
        query_start_loc_np=query_start_loc_np,
    )

    assert num_reqs_padded == len(expected) - 1
    np.testing.assert_array_equal(actual[: num_reqs_padded + 1], expected)


def test_adaptive_verification_padding_does_not_add_empty_request():
    runner = _make_runner()
    runner.max_num_reqs = 4
    query_start_loc_np = np.array([0, 1, 3, 5, 8, 8], dtype=np.int32)

    actual, num_reqs_padded = runner._pad_adaptive_query_start_loc_for_fia(
        num_tokens_padded=8,
        num_reqs_padded=4,
        num_reqs=4,
        query_start_loc_np=query_start_loc_np,
    )

    assert num_reqs_padded == 4
    np.testing.assert_array_equal(actual[:5], [0, 1, 3, 5, 8])


def test_ascend_adaptive_reallocation_builds_monotonic_offsets():
    manager = AscendAdaptiveVerificationManager.__new__(AscendAdaptiveVerificationManager)
    manager._batch_budget = (
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
        3,
    )
    manager._batch_draft_capacity = torch.empty(2, dtype=torch.int32)
    manager._num_non_draft_tokens = torch.empty(2, dtype=torch.int32)
    manager._cu_num_logits = torch.empty(3, dtype=torch.int32)
    manager.query_start_loc = torch.empty(3, dtype=torch.int32)
    manager.num_bonus_tokens = 1
    manager.num_speculative_steps = 2

    cu_num_logits, query_start_loc, draft_budget = manager.reallocate_drafts(
        ["a", "b"],
        torch.tensor([0, 1], dtype=torch.int32),
    )

    assert draft_budget == 3
    torch.testing.assert_close(
        cu_num_logits,
        torch.tensor([0, 2, 5], dtype=torch.int32),
    )
    torch.testing.assert_close(
        query_start_loc,
        torch.tensor([0, 4, 10], dtype=torch.int32),
    )


def test_ascend_adaptive_reallocation_eagerly_trims_draft_budget():
    manager = AscendAdaptiveVerificationManager.__new__(AscendAdaptiveVerificationManager)
    manager._batch_budget = (
        {"a": 2, "b": 2},
        {"a": 3, "b": 4},
        2,
    )
    manager._batch_draft_capacity = torch.empty(2, dtype=torch.int32)
    manager._num_non_draft_tokens = torch.empty(2, dtype=torch.int32)
    manager._cu_num_logits = torch.empty(3, dtype=torch.int32)
    manager.query_start_loc = torch.empty(3, dtype=torch.int32)
    manager._confidence_probs = torch.tensor([[0.9, 0.9], [0.8, 0.1]])
    manager.num_bonus_tokens = 1
    manager.num_speculative_steps = 2

    cu_num_logits, query_start_loc, draft_budget = manager.reallocate_drafts(
        ["a", "b"],
        torch.tensor([0, 1], dtype=torch.int32),
    )

    assert draft_budget == 2
    torch.testing.assert_close(
        cu_num_logits,
        torch.tensor([0, 3, 4], dtype=torch.int32),
    )
    torch.testing.assert_close(
        query_start_loc,
        torch.tensor([0, 5, 9], dtype=torch.int32),
    )


def test_model_with_context_forwards_adaptive_dspark_methods():
    original = Mock()
    wrapper = ModelWithContext(original)
    tensors = [torch.tensor([i]) for i in range(4)]

    wrapper.compute_confidence(tensors[0], tensors[1])
    wrapper.apply_markov_bias_gathered(*tensors)

    original.compute_confidence.assert_called_once_with(tensors[0], tensors[1])
    original.apply_markov_bias_gathered.assert_called_once_with(*tensors)


def test_sample_tokens_restores_replicated_draft_hidden_states():
    runner = _make_runner(need_timing=False)
    runner.is_last_pp_rank = True
    runner.speculator = SimpleNamespace(replicated_pcp=True)
    runner.use_spec_pp = False

    aux_hidden_states = [
        torch.arange(6, dtype=torch.float32).reshape(2, 3),
        torch.arange(4, dtype=torch.float32).reshape(2, 2),
    ]
    state = Mock(aux_hidden_states=aux_hidden_states)
    restored_state = object()
    state._replace.return_value = restored_state
    runner.execute_model_state = state

    target_hidden_states = object()
    restored_aux_hidden_states = torch.ones(4, 5)
    runner.pcp_manager = SimpleNamespace(
        restore_hidden_state_buffer=Mock(),
        restore_hidden_states=Mock(
            return_value=restored_aux_hidden_states,
        ),
    )
    runner.model = SimpleNamespace(
        get_mtp_target_hidden_states=lambda: target_hidden_states,
    )
    grammar_output = object()
    expected_output = object()

    with patch.object(
        GPUModelRunner,
        "sample_tokens",
        return_value=expected_output,
    ) as parent_sample_tokens:
        actual = runner.sample_tokens(grammar_output)

    assert actual is expected_output
    parent_sample_tokens.assert_called_once_with(grammar_output)
    runner.pcp_manager.restore_hidden_state_buffer.assert_called_once_with(target_hidden_states)
    restored_input = runner.pcp_manager.restore_hidden_states.call_args.args[0]
    torch.testing.assert_close(
        restored_input,
        torch.cat(aux_hidden_states, dim=-1),
    )
    state._replace.assert_called_once_with(aux_hidden_states=[restored_aux_hidden_states])
    assert runner.execute_model_state is restored_state
