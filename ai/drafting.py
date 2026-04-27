"""
SchedulerOutput 构造器 —— 用于 mock execute_model 中"第一条请求进入"的场景。

设计思路（对应 foobar.md）：
  第一条 prefill 请求 = 一个全新的 NewRequestData
                       + 空的 CachedRequestData
                       + 所有 prompt token 一次性调度
                       + 无 spec decode / encoder / finished

Usage::

    # 直接调用工厂函数
    so = build_first_prefill_so(prompt_len=128)

    # 或者作为 pytest fixture 注入（直接拿对象，不用再调函数）
    def test_something(first_prefill_so):
        assert first_prefill_so.total_num_scheduled_tokens == 128
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ── 基础依赖：懒加载 vllm 类型，不依赖 NPU ──────────────────────────────


def _SchedulerOutput():
    from vllm.v1.core.sched.output import SchedulerOutput
    return SchedulerOutput


def _NewRequestData():
    from vllm.v1.core.sched.output import NewRequestData
    return NewRequestData


def _CachedRequestData():
    from vllm.v1.core.sched.output import CachedRequestData
    return CachedRequestData


# ── 核心工厂函数 ─────────────────────────────────────────────────────────


def build_first_prefill_so(
    req_id: str = "req_0",
    prompt_len: int = 128,
    prefill_token_ids: list[int] | None = None,
) -> MagicMock:
    """构造第一个 prefill 请求的 SchedulerOutput。

    场景刻画:
    - 一个全新请求刚刚到达 scheduler
    - 所有 prompt token 被一次性调度出去（纯 prefill，非 chunked）
    - num_computed_tokens == 0（尚未计算任何 token）
    - 没有任何 cached request、spec decode、finished request
    - eager 模式（最简单，无 graph）
    - chunked_prefill = False => 走纯 PrefillNoCache 路径

    Args:
        req_id:             请求 ID。
        prompt_len:         prompt 总长度（即本次调度的 token 数）。
        prefill_token_ids:  可选的显式 token ID 列表，默认 1..prompt_len。

    Returns:
        MagicMock 实例，spec 绑定到 vllm 的 SchedulerOutput 类。
    """
    so = MagicMock(spec=_SchedulerOutput())

    if prefill_token_ids is None:
        prefill_token_ids = list(range(1, prompt_len + 1))

    # ── 1. scheduled_new_reqs：那个唯一的请求 ──────────────────────────
    new_req = MagicMock(spec=_NewRequestData())
    new_req.req_id = req_id
    new_req.prompt_token_ids = prefill_token_ids[:prompt_len]
    new_req.prefill_token_ids = prefill_token_ids  # v2 特有
    new_req.num_computed_tokens = 0  # ★ 关键：还什么都没算
    new_req.block_ids = ([0],)  # 1 个 KV cache group，已分配 block 0
    new_req.mm_features = []
    new_req.sampling_params = MagicMock()
    new_req.pooling_params = None
    new_req.lora_request = None

    # ── 2. scheduled_cached_reqs：空的（没有任何 cached 请求） ──────────
    cached = MagicMock(spec=_CachedRequestData())
    cached.req_ids = []
    cached.resumed_req_ids = set()
    cached.new_token_ids = []
    cached.all_token_ids = {}
    cached.new_block_ids = []
    cached.num_computed_tokens = []
    cached.num_output_tokens = []

    # ── 3. 填充 SchedulerOutput 字段 ──────────────────────────────────
    so.scheduled_new_reqs = [new_req]
    so.scheduled_cached_reqs = cached
    so.num_scheduled_tokens = {req_id: prompt_len}
    so.total_num_scheduled_tokens = prompt_len
    so.scheduled_spec_decode_tokens = {}
    so.scheduled_encoder_inputs = {}
    so.num_common_prefix_blocks = []
    so.finished_req_ids = set()
    so.free_encoder_mm_hashes = []
    so.preempted_req_ids = None
    so.has_structured_output_requests = False
    so.pending_structured_output_tokens = False
    so.num_invalid_spec_tokens = None
    so.kv_connector_metadata = None
    so.ec_connector_metadata = None
    so.new_block_ids_to_zero = None

    return so


# ── pytest fixture ────────────────────────────────────────────────────────


@pytest.fixture
def first_prefill_so() -> MagicMock:
    """返回一个已构造好的 SchedulerOutput（默认 128 token 的第一个 prefill）。

    用法::

        def test_prefill_model(first_prefill_so):
            assert first_prefill_so.total_num_scheduled_tokens == 128

    如果想自定义参数，直接用 build_first_prefill_so 函数::

        def test_custom(first_prefill_so):
            so = build_first_prefill_so(prompt_len=256)
    """
    return build_first_prefill_so()
