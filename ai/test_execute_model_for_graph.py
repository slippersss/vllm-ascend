import inspect
import ast
import textwrap


from unittest.mock import MagicMock, patch, Mock
import unittest
import pytest
import torch
from contextlib import ExitStack
import random
import inspect
import numpy as np

from vllm_ascend.worker.v2.attn_utils import get_attn_mask_builder, build_attn_metadata, build_attn_state, _get_layer_kv_cache_specs, _get_attention_kv_cache_dims, _align_memory, _allocate_kv_cache, _reshape_kv_cache
import vllm_ascend.worker.v2.attn_utils
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from types import MethodType
from contextlib import nullcontext

# FULL_DECODE_ONLY ASSUMPTION


def get_param_names(sig):
    # sig = inspect.signature(fn)
    return [p.name for p in sig.parameters.values()]


from vllm_ascend.worker.v2.model_runner import NPUModelRunner
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.core.sched.output import CachedRequestData
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

from vllm_ascend.worker.v2.model_states.default import AscendModelState


from vllm.v1.worker.utils import AttentionGroup
from vllm_ascend.attention.attention_v1 import AscendAttentionBackend


from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager

from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import AscendMetadata
from vllm_ascend.worker.v2.input_batch import AscendInputBatch


from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor, KVQuantMode
)
from vllm.v1.worker.gpu.kv_connector import KVConnector
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.attention.attention_v1 import AscendAttentionBackend, AscendAttentionMetadataBuilder
from vllm_ascend.attention.mla_v1 import AscendMLABackend, AscendMLAMetadataBuilder
from vllm_ascend.patch.platform.patch_kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager
from vllm_ascend.worker.v2.block_table import AscendBlockTables
from vllm_ascend.worker.v2.input_batch import AscendInputBuffers
from vllm_ascend.worker.v2.states import AscendRequestState

class TestExecuteModelForGraph:
    @pytest.fixture(scope="class", autouse=True)
    def prerequisite(self):
        # is_encoder_decoder

        assert hasattr(NPUModelRunner, "finish_requests")
        assert hasattr(NPUModelRunner, "free_states")
        assert hasattr(NPUModelRunner, "add_requests")
        assert hasattr(NPUModelRunner, "update_requests")

        # block_tables
        assert hasattr(AscendBlockTables, "apply_staged_writes")

        # kv_connector
        assert hasattr(KVConnector, "no_forward")

        # cudagraph_manager
        assert hasattr(ModelAclGraphManager, "dispatch")

        # req_states

        # input_buffers

    @pytest.fixture
    def kv_cache_config_attn_groups_qwen3_06b(self):
        num_layers = 28

        num_blocks = 666
        kv_cache_tensors = []
        for i in range(num_layers):
            kv_cache_tensors.append(
                KVCacheTensor(
                    # (kv) * num_blocks * block_size * num_heads * head_size * (bfloat16)
                    size=2 * num_blocks * 128 * 8 * 128 * 2, shared_by=[f"model.layers.{i}.self_attn.attn"]
                )
            )
        layer_names = []
        for i in range(num_layers):
            layer_names.append(f"model.layers.{i}.self_attn.attn")
        kv_cache_groups = [
            KVCacheGroupSpec(
                layer_names=layer_names,
                kv_cache_spec=FullAttentionSpec(
                    block_size=128,
                    num_kv_heads=8,
                    head_size=128,
                    dtype=torch.bfloat16,
                    kv_quant_mode=KVQuantMode.NONE,
                    page_size_padded=None,
                    head_size_v=128,
                    sliding_window=None,
                    attention_chunk_size=None,
                ),
            )
        ]

        kv_cache_config = KVCacheConfig(
            num_blocks=num_blocks, kv_cache_tensors=kv_cache_tensors, kv_cache_groups=kv_cache_groups
        )

        layer_names = []
        for i in range(num_layers):
            layer_names.append(f"model.layers.{i}.self_attn.attn")
        mock_metadata_builder = MagicMock(spec=AscendAttentionMetadataBuilder)
        attn_groups = [
            [
                AttentionGroup(
                    backend=AscendAttentionBackend,
                    layer_names=layer_names,
                    kv_cache_spec=FullAttentionSpec(
                        block_size=128,
                        num_kv_heads=8,
                        head_size=128,
                        dtype=torch.bfloat16,
                        kv_quant_mode=KVQuantMode.NONE,
                        page_size_padded=None,
                        head_size_v=128,
                        sliding_window=None,
                        attention_chunk_size=None,
                    ),
                    kv_cache_group_id=0,
                    # mock it in test
                    metadata_builders=[mock_metadata_builder],
                )
            ]
        ]

        return kv_cache_config, attn_groups, mock_metadata_builder

    @pytest.fixture
    def kv_cache_config_attn_groups_dsv2_lite(self):
        num_layers = 27

        num_blocks = 666
        kv_cache_tensors = []
        for i in range(num_layers):
            kv_cache_tensors.append(
                KVCacheTensor(
                    # num_blocks * block_size * num_heads * head_size (kv) * (bfloat16)
                    size=num_blocks * 128 * 1 * (512 + 64) * 2, shared_by=[f"model.layers.{i}.self_attn.attn"]
                )
            )
        layer_names = []
        for i in range(num_layers):
            layer_names.append(f"model.layers.{i}.self_attn.attn")
        kv_cache_groups = [
            KVCacheGroupSpec(
                layer_names=layer_names,
                kv_cache_spec=AscendMLAAttentionSpec(
                    block_size=128,
                    num_kv_heads=1,
                    head_size=576,
                    dtype=torch.bfloat16,
                    kv_quant_mode=KVQuantMode.NONE,
                    page_size_padded=None,
                    head_size_v=576,
                    sliding_window=None,
                    attention_chunk_size=None,
                    cache_dtype_str="auto",
                    sparse_head_dim=None,
                    cache_sparse_c8=False,
                    c8_k_cache_dtype=torch.int8,
                    c8_k_scale_cache_dtype=torch.float16,
                ),
            )
        ]

        kv_cache_config = KVCacheConfig(
            num_blocks=num_blocks, kv_cache_tensors=kv_cache_tensors, kv_cache_groups=kv_cache_groups
        )

        layer_names = []
        for i in range(num_layers):
            layer_names.append(f"model.layers.{i}.self_attn.attn")
        mock_metadata_builder = MagicMock(spec=AscendMLAMetadataBuilder)
        attn_groups = [
            [
                AttentionGroup(
                    backend=AscendMLABackend,
                    layer_names=layer_names,
                    kv_cache_spec=AscendMLAAttentionSpec(
                        block_size=128,
                        num_kv_heads=1,
                        head_size=576,
                        dtype=torch.bfloat16,
                        kv_quant_mode=KVQuantMode.NONE,
                        page_size_padded=None,
                        head_size_v=576,
                        sliding_window=None,
                        attention_chunk_size=None,
                        cache_dtype_str="auto",
                        sparse_head_dim=None,
                        cache_sparse_c8=False,
                        c8_k_cache_dtype=torch.int8,
                        c8_k_scale_cache_dtype=torch.float16,
                    ),
                    kv_cache_group_id=0,
                    # mock it in test
                    metadata_builders=[mock_metadata_builder],
                )
            ]
        ]

        return kv_cache_config, attn_groups, mock_metadata_builder

    @pytest.fixture
    def scheduler_output_one_prefill(self):
        new_request_data = NewRequestData(
            req_id="req_example",
            prompt_token_ids=list(range(262144, 262144 + 32)),
            mm_features=[],
            sampling_params=None,  # not important
            pooling_params=None,
            block_ids=([1],),
            num_computed_tokens=0,
            lora_request=None,
            prefill_token_ids=list(range(262144, 262144 + 32)),
        )
        cached_request_data = CachedRequestData(
            req_ids=[],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
        )

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[new_request_data],
            scheduled_cached_reqs=cached_request_data,
            num_scheduled_tokens={"req_example": 32},
            total_num_scheduled_tokens=32,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],  # not important
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            preempted_req_ids=None,
            has_structured_output_requests=False,
            pending_structured_output_tokens=False,
            num_invalid_spec_tokens=None,
            kv_connector_metadata=None,
            ec_connector_metadata=None,
            new_block_ids_to_zero=None,
        )

        return scheduler_output

    @pytest.fixture
    def scheduler_output_one_decode(self):
        cached_request_data = CachedRequestData(
            req_ids=["req_example"],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[None],
            num_computed_tokens=[32],
            num_output_tokens=[1],
        )

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=cached_request_data,
            num_scheduled_tokens={"req_example": 1},
            total_num_scheduled_tokens=1,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],  # not important
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            preempted_req_ids=None,
            has_structured_output_requests=False,
            pending_structured_output_tokens=False,
            num_invalid_spec_tokens=None,
            kv_connector_metadata=None,
            ec_connector_metadata=None,
            new_block_ids_to_zero=None,
        )

        return scheduler_output

    @pytest.fixture
    def model_runner_basic_setting(self, request):
        max_num_reqs = max_num_seqs =  256
        max_num_tokens = max_model_len = 32768
        cudagraph_capture_sizes = [1, 2, 4, 8, 16, 32, 64]

        mr = MagicMock(spec=NPUModelRunner)

        mr.is_encoder_decoder = False
        mr.dp_size = 1
        mr.dp_rank = 0
        mr.device = torch.device("cpu")
        mr.vllm_config = MagicMock(spec=VllmConfig)
        mr.lora_config = None
        mr.max_num_reqs = max_num_reqs
        mr.supports_mm_inputs = False
        mr.is_first_pp_rank = True
        mr.is_last_pp_rank = True
        mr.use_aux_hidden_state_outputs = False
        mr.model = MagicMock(return_value=torch.zeros(10, dtype=torch.bfloat16))

        kv_cache_config, attn_groups, mock_builder = request.getfixturevalue("kv_cache_config_attn_groups_qwen3_06b")
        mr.kv_cache_config = kv_cache_config
        mr.attn_groups = attn_groups

        mr.finish_requests = MagicMock(return_value=None)
        mr.free_states = MagicMock(return_value=None)
        mr.add_requests = MagicMock(return_value=None)
        mr.update_requests = MagicMock(return_value=None)

        mr._update_seq_lens_cpu = MagicMock(return_value=None)
        mr.prepare_attn = MagicMock(return_value=None)

        mr.block_tables = MagicMock(spec=AscendBlockTables)
        mr.block_tables.apply_staged_writes = MagicMock(return_value=None)
        mr.block_tables.block_tables = [torch.zeros((max_num_reqs, 666), dtype=torch.int32)]

        mr.kv_connector = MagicMock(spec=KVConnector)
        mr.kv_connector.no_forward = MagicMock(return_value=MagicMock())

        mr.cudagraph_manager = MagicMock(spec=ModelAclGraphManager)
        mr.cudagraph_manager._graphs_captured = True
        mr.cudagraph_manager._candidates = []
        for i in range(len(cudagraph_capture_sizes)):
            start = cudagraph_capture_sizes[i - 1] + 1 if i > 0 else 0
            end = cudagraph_capture_sizes[i] + 1
            for j in range(start, end):
                mr.cudagraph_manager._candidates.append(
                    [
                        BatchExecutionDescriptor(
                            cg_mode=CUDAGraphMode.FULL,
                            num_tokens=cudagraph_capture_sizes[i],
                            num_reqs=cudagraph_capture_sizes[i],
                            uniform_token_count=1,
                        )
                    ]
                )
        mr.cudagraph_manager.dispatch = MethodType(ModelAclGraphManager.dispatch, mr.cudagraph_manager)

        mr.req_states = MagicMock(spec=AscendRequestState)
        mr.req_states.req_id_to_index = {}  # should set in case?
        # mr.req_states.any_prefills
        mr.req_states.next_prefill_tokens = MagicMock()
        mr.req_states.all_token_ids = MagicMock()
        mr.req_states.prefill_len = MagicMock()
        mr.req_states.num_computed_tokens = MagicMock()
        mr.req_states.last_sampled_tokens = MagicMock()
        mr.req_states.prefill_len = MagicMock()
        mr.req_states.draft_tokens = MagicMock()

        mr.input_buffers = MagicMock(spec=AscendInputBuffers)
        mr.input_buffers.seq_lens_np = MagicMock()
        mr.input_buffers.query_start_loc = torch.zeros(
            max_num_reqs + 2,
            dtype=torch.int32,
        )
        mr.input_buffers.seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32)
        mr.input_buffers.seq_lens_cpu = torch.zeros(
            max_num_reqs,
            dtype=torch.int32
        )
        mr.input_buffers.seq_lens_np = mr.input_buffers.seq_lens_cpu.numpy()
        mr.input_buffers.input_ids = torch.zeros(max_num_tokens, dtype=torch.int32)
        mr.input_buffers.positions = torch.zeros(max_num_tokens, dtype=torch.int64)

        mr.model_state = MagicMock(spec=AscendModelState)
        mr.model_state.prepare_attn = MethodType(AscendModelState.prepare_attn, mr.model_state)
        mr.model_state.max_model_len = max_model_len

        mock_builder.decode_threshold = 1
        mock_builder.kv_cache_spec = None
        mock_builder.speculative_config = None
        mock_builder.model_config = MagicMock()
        mock_builder.model_config.hf_text_config = None
        mock_builder.model_config.runner_type = None
        mock_builder.device = torch.device("cpu")
        mock_builder.attn_mask_builder = AttentionMaskBuilder(torch.device("cpu"))

        from vllm_ascend.attention.attention_v1 import AscendAttentionMetadataBuilder
        # builder = MagicMock()
        mock_builder.build = MethodType(AscendAttentionMetadataBuilder.build, mock_builder)
        # mr.attn_groups[0][0].get_metadata_builder = MagicMock(return_value=builder)

        mr.prepare_inputs = MethodType(NPUModelRunner.prepare_inputs, mr)

        # You need to mock it by yourself
        # with patch("vllm_ascend.worker.v2.model_runner.build_attn_state") as mock_build_attn_state:
            # mock_build_attn_state.return_value
            # combine_sampled_and_draft_tokens

        slot_mappings_by_layer = {}
        for layer_name in kv_cache_config.kv_cache_groups[0].layer_names:
            slot_mappings_by_layer[layer_name] = torch.arange(128, 160)

        mr.execute_model = MethodType(NPUModelRunner.execute_model, mr)

        # def mock_prepare_attn(self, input_batch):
        ret_0 = (mr.block_tables.block_tables[0][:1],)
        ret_1 = torch.arange(128, 160).reshape(1, -1).to(torch.int32)
        # return ret_0, ret_1
        mr.prepare_attn = MagicMock(return_value=(ret_0, ret_1))

        with (
            patch("vllm_ascend.worker.v2.model_runner.prepare_pos_seq_lens") as mock_0,
            patch("vllm_ascend.worker.v2.model_runner.update_cos_sin") as mock_1,
            patch("vllm.v1.worker.gpu.model_runner.build_slot_mappings_by_layer") as mock_2,
        ):
            mock_0.return_value = None
            mock_1.return_value = None
            mock_2.return_value = slot_mappings_by_layer

            yield mr

    @pytest.fixture
    def model_runner_one_prefill(self, request):
        mr = request.getfixturevalue("model_runner_basic_setting")

        mr.req_states.req_id_to_index["req_example"] = 0
        mr.req_states.any_prefills = MagicMock(return_value=True)

        with (
            patch("vllm_ascend.worker.v2.model_runner.build_attn_state") as mock_0,
            patch("vllm_ascend.worker.v2.model_runner.combine_sampled_and_draft_tokens") as mock_1,
            patch("vllm.v1.worker.gpu.model_runner.set_forward_context"),
            patch("torch.Tensor.pin_memory", new=lambda self: self),
            patch("vllm_ascend.worker.v2.model_runner.prepare_prefill_inputs"),
        ):
            mock_0.return_value = AscendAttentionState.PrefillNoCache
            mock_1.return_value = torch.tensor([31], dtype=torch.int64)

            yield mr

    def test_data(self, request):
        # kv_cache_config, attn_groups, mock_builder = request.getfixturevalue("kv_cache_config_attn_groups_qwen3_06b")
        # print(f"{kv_cache_config=}")
        # print(f"{attn_groups=}")

        # kv_cache_config, attn_groups, mock_builder = request.getfixturevalue("kv_cache_config_attn_groups_dsv2_lite")
        # print(f"{kv_cache_config=}")
        # print(f"{attn_groups=}")

        # scheduler_output = request.getfixturevalue("scheduler_output_one_prefill")
        # print(f"{scheduler_output=}")

        # scheduler_output = request.getfixturevalue("scheduler_output_one_decode")
        # print(f"{scheduler_output=}")

        mr = request.getfixturevalue("model_runner_one_prefill")
        so = request.getfixturevalue("scheduler_output_one_prefill")
        # print(f"{mr.cudagraph_manager._candidates=}")
        mr.execute_model(so)
        print(f"{mr.execute_model_state=}")

class Nope():
    @pytest.fixture(autouse=True, scope="class")
    def prerequisite(self):
        pass

    @pytest.fixture
    def scheduler_output_one_prefill(self):
        new_request_data = NewRequestData(
            req_id="req_example",
            prompt_token_ids=list(range(262144, 262144 + 32)),
            mm_features=[],
            sampling_params=None,  # not important in this case
            pooling_params=None,
            block_ids=([1],),
            num_computed_tokens=0,
            lora_request=None,
            prefill_token_ids=list(range(262144, 262144 + 32)),
        )

        cached_request_data = CachedRequestData(
            req_ids=[],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
        )

        so = SchedulerOutput(
            scheduled_new_reqs=[new_request_data],
            scheduled_cached_reqs=cached_request_data,
            num_scheduled_tokens={"req_example": 32},
            total_num_scheduled_tokens=32,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],  # not important in this case
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            preempted_req_ids=None,
            has_structured_output_requests=False,
            pending_structured_output_tokens=False,
            num_invalid_spec_tokens=None,
            kv_connector_metadata=None,
            ec_connector_metadata=None,
            new_block_ids_to_zero=None,
        )

        return so

    @pytest.fixture
    def attn_metadata_one_prefill(self):
        pass

    @pytest.fixture
    def scheduler_output_one_decode(self):
        cached_request_data = CachedRequestData(
            req_ids=["req_example"],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[None],
            num_computed_tokens=[32],
            num_output_tokens=[1],
        )

        so = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=cached_request_data,
            num_scheduled_tokens={"req_example": 1},
            total_num_scheduled_tokens=1,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],  # not important in this case
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            preempted_req_ids=None,
            has_structured_output_requests=False,
            pending_structured_output_tokens=False,
            num_invalid_spec_tokens=None,
            kv_connector_metadata=None,
            ec_connector_metadata=None,
            new_block_ids_to_zero=None,
        )

        return so

    def test_data(self, request):
        so_one_prefill = request.getfixturevalue("scheduler_output_one_prefill")
        print(f"{so_one_prefill=}")
        so_one_decode = request.getfixturevalue("scheduler_output_one_decode")
        print(f"{so_one_decode=}")
        pass

    @pytest.fixture
    def qwen3_06b_input_another(self):
        import vllm.v1.kv_cache_interface
        import vllm_ascend.attention.mla_v1
        num_layers = 28

        kv_cache_config = MagicMock()
        kv_cache_config.num_blocks = 512   # you can set it in any way
        kv_cache_config.kv_cache_tensors = [MagicMock() for _ in range(num_layers)]
        kv_cache_config.kv_cache_groups = [MagicMock()] # maybe we can have multiple groups, this is just simulated dsv2 lite currently

        for i, item in enumerate(kv_cache_config.kv_cache_tensors):
            item.size = 2 * 512 * 128 * 8 * 128 * 2  # 2 (k/v respectively) * num_blocks * block_size * num_heads * head_size * dtype in Bytes
            item.shared_by = [f"model.layers.{i}.self_attn.attn"]

        mock_layer_spec = {}


        kv_cache_group_spec = kv_cache_config.kv_cache_groups[0]

        kv_cache_group_spec.layer_names = []
        for i in range(num_layers):
            kv_cache_group_spec.layer_names.append(f"model.layers.{i}.self_attn.attn")

        kv_cache_group_spec.kv_cache_spec = MagicMock(spec=vllm.v1.kv_cache_interface.FullAttentionSpec)
        kv_cache_group_spec.kv_cache_spec.block_size = 128
        kv_cache_group_spec.kv_cache_spec.num_kv_heads = 8
        kv_cache_group_spec.kv_cache_spec.head_size = 128
        kv_cache_group_spec.kv_cache_spec.dtype = torch.bfloat16
        kv_cache_group_spec.kv_cache_spec.page_size_padded = None
        kv_cache_group_spec.kv_cache_spec.head_size_v = 128
        kv_cache_group_spec.kv_cache_spec.page_size_bytes = 2 * 128 * 8 * 128 * 2
        # kv_cache_group_spec.kv_cache_spec.sliding_window = None
        # kv_cache_group_spec.kv_cache_spec.attention_chunk_

        for i in range(num_layers):
            mock_layer_spec[f"model.layers.{i}.self_attn.attn"] = kv_cache_group_spec.kv_cache_spec

        kv_cache_raw_tensors = {}
        for i in range(num_layers):
            name = f"model.layers.{i}.self_attn.attn"
            kv_cache_raw_tensors[name] = (
                torch.zeros(512 * 128 * 8 * 128 * 2, dtype=torch.int8), # 512 kv_lora_rank
                torch.zeros(512 * 128 * 8 * 128 * 2, dtype=torch.int8), # 64 qk_rope_head_dim
            )

        attn_backends = {}
        for i in range(num_layers):
            name = f"model.layers.{i}.self_attn.attn"
            attn_backends[name] = MagicMock(spec=vllm_ascend.attention.attention_v1.AscendAttentionBackend)
            attn_backends[name].get_kv_cache_shape = lambda a, b, c, d, e: (2, a, b, c, d)

        vllm_config = MagicMock()
        vllm_config.kv_transfer_config = None
        
        with patch("vllm_ascend.worker.v2.attn_utils.get_current_vllm_config", new=lambda: vllm_config), \
             patch("vllm_ascend.worker.v2.attn_utils._get_attention_kv_cache_dims", new=lambda a, b: (128, 128)), \
             patch("vllm_ascend.worker.v2.attn_utils._get_layer_kv_cache_specs", new=lambda _: mock_layer_spec):
            yield kv_cache_config, kv_cache_raw_tensors, attn_backends


    @pytest.fixture
    def model_runner_basic_setting(self, request):
        max_num_reqs = 256
        max_model_len = 32768
        max_num_tokens = 32768
        num_blocks = 32
        mr = MagicMock(spec=NPUModelRunner)

        ################################################################################################################
        mr.finish_requests = MagicMock(return_value=None)  # just delete some finished/preempted reqs
        mr.free_states = MagicMock(return_value=None)  # just delete encoder_cache, nothing else, safe
        mr.add_requests = MagicMock(return_value=None)
        mr.update_requests = MagicMock(return_value=None)  # not care about another thing in one prefill scenario
        ################################################################################################################

        # we don't care req_states, related output should be explicitly construct for the specific input
        mr.req_states = MagicMock(spec=AscendRequestState)
        mr.req_states.req_id_to_index = {}
        mr.req_states.any_prefills = MagicMock(return_value=True)
        mr.req_states.next_prefill_tokens = MagicMock()
        mr.req_states.all_token_ids = MagicMock()
        mr.req_states.prefill_len = MagicMock()
        mr.req_states.num_computed_tokens = MagicMock()
        mr.req_states.last_sampled_tokens = MagicMock()
        mr.req_states.prefill_len = MagicMock()
        mr.req_states.draft_tokens = MagicMock()

        # as for block table, so actually I am thinking that maybe we don't need it, just fake it
        mr.block_tables = MagicMock(spec=AscendBlockTables)
        mr.block_tables.apply_staged_writes = MagicMock(return_value=None)
        mr.block_tables.block_tables = [torch.zeros((max_num_reqs, num_blocks), dtype=torch.int32)]

        # input buffers is actually for input batch, maybe?
        mr.input_buffers = MagicMock(spec=AscendInputBuffers)
        mr.input_buffers.query_start_loc = torch.zeros(max_num_reqs + 2, dtype=torch.int32)
        mr.input_buffers.input_ids = torch.zeros(max_num_tokens, dtype=torch.int32)
        mr.input_buffers.seq_lens_cpu = torch.zeros(max_num_reqs, dtype=torch.int32)
        mr.input_buffers.seq_lens_np = mr.input_buffers.seq_lens_cpu.numpy()
        mr.input_buffers.seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device=torch.device("cpu"))
        mr.input_buffers.positions = torch.zeros(max_num_tokens, dtype=torch.int64)

        # about model_state, construce some important class that we care, about model_state
        mr.model_state = MagicMock(spec=AscendModelState)
        mr.model_state.prepare_attn = MethodType(AscendModelState.prepare_attn, mr.model_state)
        mr.model_state.max_model_len = 65536

        # we extend the test to dispatch, we care about it
        mr.cudagraph_manager = MagicMock(spec=ModelAclGraphManager)
        mr.cudagraph_manager._graphs_captured = True
        mr.cudagraph_manager._candidates = [] # decode only
        # capture sizes = [1, 2, 4, 8, 16, 32, 64]
        sizes = [1, 2, 4, 8, 16, 32, 64]
        start_end_group = []
        start_end_group.append([0, sizes[0] + 1])
        for i in range(1, len(sizes)):
            start = sizes[i - 1] + 1
            end = sizes[i] + 1
            start_end_group.append([start, end])
        for group in start_end_group:
            for i in range(group[0], group[1]):
                desc = BatchExecutionDescriptor(
                    cg_mode=CUDAGraphMode.FULL,
                    num_tokens=group[1] - 1,
                    num_reqs=group[1] - 1,
                    uniform_token_count=1,
                )
                mr.cudagraph_manager._candidates.append([desc])
        mr.cudagraph_manager.dispatch = MethodType(ModelAclGraphManager.dispatch, mr.cudagraph_manager)  # we care about it 

        # attn_groups
        mr.attn_groups = [[]]
        mr.attn_groups[0].append(AttentionGroup(
            backend=AscendAttentionBackend,
            layer_names=[f"model.layers.{i}.self_attn.attn" for i in range(28)],
            kv_cache_spec=None,
            kv_cache_group_id=None,
        ))

        # about the builder
        builder = MagicMock()
        builder.decode_threshold = 1
        builder.kv_cache_spec = None
        builder.speculative_config = None
        builder.model_config = MagicMock()
        builder.model_config.hf_text_config = None
        builder.model_config.runner_type = None
        builder.device = torch.device("cpu")
        builder.attn_mask_builder = AttentionMaskBuilder(torch.device("cpu"))

        from vllm_ascend.attention.attention_v1 import AscendAttentionMetadataBuilder
        # builder = MagicMock()
        builder.build = MethodType(AscendAttentionMetadataBuilder.build, builder)
        mr.attn_groups[0][0].get_metadata_builder = MagicMock(return_value=builder)

        yield mr

    """
    construct for exactly one prefill
    """
    @pytest.fixture
    def model_runner_one_prefill(self, request):
        mr = request.getfixturevalue("model_runner_basic_setting")

        # ir-releated self.method
        # you can mock it, but they should promise that they are correct!

        # overall state maintaining
        mr.req_states.req_id_to_index["req_example"] = 0
        new_block_ids = ([1],)
        mr.block_tables.block_tables[0][0][:len(new_block_ids[0])] = torch.tensor(new_block_ids[0], dtype=torch.int32)  # unnecessary actually, not in my route
        # ------------------------------------------------------------------------------------

        def mock_update_seq_lens_cpu(self, scheduler_output, req_ids):
            mr.input_buffers.seq_lens_cpu[0] = 32
        mr._update_seq_lens_cpu = MethodType(mock_update_seq_lens_cpu, mr) # input_buffer seq_lens_cpu will be affacted, we don;t care about it currently, actually we should update it

        # according to specific input, you can adjust it
        def mock_prepare_attn(self, input_batch):
            ret_0 = (mr.block_tables.block_tables[0][:1],)
            ret_1 = torch.arange(128, 160).reshape(1, -1).to(torch.int32)
            return ret_0, ret_1
        mr.prepare_attn = MethodType(mock_prepare_attn, mr)

        # basic variables, nothing to do with my cases
        mr.is_encoder_decoder = False
        mr.dp_size = 1
        mr.dp_rank = 0
        mr.vllm_config = MagicMock() # not important
        mr.device = torch.device("cpu")
        mr.lora_config = None
        mr.supports_mm_inputs = False
        mr.is_first_pp_rank = True
        mr.use_aux_hidden_state_outputs = False
        mr.is_last_pp_rank = True
        mr.use_aux_hidden_states = False
        mr.max_num_reqs = 256
        mr.use_dcp = False
        mr.rope_state = None
        mr.kv_connector = MagicMock()
        mr.model = MagicMock(return_value=torch.zeros(10, dtype=torch.bfloat16))

        kv_cache_config, _, _ = request.getfixturevalue("qwen3_06b_input_another")
        mr.kv_cache_config = kv_cache_config

        # The key method we want to check
        # bind those methods that we care, importantt
        mr.execute_model = MethodType(NPUModelRunner.execute_model, mr)
        mr.prepare_inputs = MethodType(NPUModelRunner.prepare_inputs, mr) # is this ok? not sure about it

        # external dependency
        def mock_prepare_prefill_inputs(
            input_ids,
            next_prefill_tokens,
            idx_mapping,
            query_start_loc,
            all_token_ids,
            prefill_len,
            num_computed_tokens,
        ):
            input_ids = torch.tensor(list(range(262144, 262144 + 32)))
            mr.input_buffers.input_ids[:input_ids.shape[0]] = input_ids

        def mock_prepare_pos_seq_lens(
            idx_mapping,
            query_start_loc,
            num_computed_tokens,
            pos,
            seq_lens,
        ):
            # mr.input_buffers.input_ids[:64] = torch.arange(0, 64)
            mr.input_buffers.seq_lens[0] = 32
            mr.input_buffers.positions[:32] = torch.arange(32)

        def mock_build_slot_mappings_by_layer(slot_mappings, kv_cache_config):
            temp = {}
            for layer_name in kv_cache_config.kv_cache_groups[0].layer_names:
                temp[layer_name] = torch.arange(128, 160)
            return temp

        def fake_pin(self):
            return self

        with (
            patch("vllm_ascend.worker.v2.model_runner.prepare_prefill_inputs", new=mock_prepare_prefill_inputs),
            patch("vllm_ascend.worker.v2.model_runner.prepare_pos_seq_lens", new=mock_prepare_pos_seq_lens),
            patch("vllm_ascend.worker.v2.model_runner.combine_sampled_and_draft_tokens") as mock_2,
            patch("vllm_ascend.worker.v2.model_runner.update_cos_sin") as mock_3,
            patch("vllm.v1.worker.gpu.model_runner.build_slot_mappings_by_layer", new=mock_build_slot_mappings_by_layer),
            patch("vllm.v1.worker.gpu.model_runner.set_forward_context") as mock_4,
            patch("torch.Tensor.pin_memory", new=fake_pin),
            patch("vllm_ascend.worker.v2.model_runner.build_attn_state") as mock_6,
        ):
            mock_2.return_value = torch.tensor([31]) # not important
            mock_3.return_value = None

            mock_6.return_value = AscendAttentionState.PrefillNoCache
            yield mr

    def test_execute_model_one_prefill(self, request):
        scheduler_output = request.getfixturevalue("scheduler_output_one_prefill")
        model_runner = request.getfixturevalue("model_runner_one_prefill")
        model_runner.execute_model(scheduler_output=scheduler_output)
        # need assert here
        # then we just need to assert final execution_state
        # incude attn_metadata, input_batch, slot_mapping_by layer
        block_tables = torch.zeros((1, 32), dtype=torch.int32)
        block_tables[0][0] = 1
        query_start_loc = torch.tensor([0, 32], dtype=torch.int32)
        seq_lens = torch.tensor([32], dtype=torch.int32)
        seq_lens_cpu = torch.tensor([32], dtype=torch.int32)
        seq_lens_list = [32]
        max_query_len = 32
        actual_seq_lengths_q = [32]
        slot_mapping = torch.arange(128, 160).to(torch.int32)

        attn_mask = torch.triu(torch.ones(2048, 2048), diagonal=1).to(torch.int8)

        answer = AscendMetadata(
            num_actual_tokens=32,
            num_decode_tokens=0,
            block_tables=block_tables,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens,
            seq_lens_list=seq_lens_list,
            max_query_len=max_query_len,
            actual_seq_lengths_q=actual_seq_lengths_q,
            slot_mapping=slot_mapping,
            attn_mask=attn_mask,
            # swa_mask=None,
            attn_state=AscendAttentionState.PrefillNoCache,
            num_prefills=1,
            num_decodes=0,
            causal=True,
            model_runner_type=None,
        )
        print(f"{model_runner.model_state.attn_metadata['model.layers.7.self_attn.attn'].__dataclass_fields__.keys()=}")
        print(f"{answer.__dataclass_fields__.keys()=}")

        # attn_metadata checking
        assert model_runner.model_state.attn_metadata['model.layers.7.self_attn.attn'].__dataclass_fields__.keys() == answer.__dataclass_fields__.keys()
        for i in range(28):
            layer_name = f"model.layers.{i}.self_attn.attn"
            for key in answer.__dataclass_fields__.keys():
                print(layer_name, key)
                attr1 = getattr(model_runner.model_state.attn_metadata[layer_name], key)
                attr2 = getattr(answer, key)
                print(f"{attr1=}")
                print(f"{attr2=}")
                if isinstance(attr1, torch.Tensor):
                    assert torch.equal(attr1, attr2)
                # elif isinstance(attr2)
                else:
                    assert attr1 == attr2

        # input_batch checking
        input_batch = model_runner.execute_model_state.input_batch
        seq_lens_np = np.zeros(256, dtype=np.int32)
        seq_lens_np[0] = 32
        answer = AscendInputBatch(
            req_ids=["req_example"],
            num_reqs=1,
            num_reqs_after_padding=1,
            idx_mapping=torch.tensor([0]),
            idx_mapping_np=np.array([0]),
            expanded_idx_mapping=torch.tensor([0]),
            expanded_local_pos=torch.tensor([0]),
            num_scheduled_tokens=np.array([32]),
            num_tokens=32,
            num_tokens_after_padding=32,
            num_draft_tokens=0,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc.numpy(),
            seq_lens=seq_lens,
            dcp_local_seq_lens=None,  # TODO(Ronald1995): support cp.
            input_ids=torch.arange(262144, 262144 + 32),
            positions=torch.arange(0, 32),
            logits_indices=torch.tensor([31]),
            cu_num_logits=torch.tensor([0, 1]),
            cu_num_logits_np=np.array([0, 1]),
            has_structured_output_reqs=scheduler_output.has_structured_output_requests,
            # extra attributes for ascend npus.
            seq_lens_np=seq_lens_np,
            attn_state=AscendAttentionState.PrefillNoCache,
        )

        print(f"*" * 50)
        assert input_batch.__dataclass_fields__.keys() == answer.__dataclass_fields__.keys()
        for key in answer.__dataclass_fields__.keys():
            print(key)
            attr1 = getattr(input_batch, key)
            attr2 = getattr(answer, key)
            print(f"{attr1=}")
            print(f"{attr2=}")
            if isinstance(attr1, torch.Tensor):
                assert torch.equal(attr1, attr2)
            # elif isinstance(attr2)
            elif isinstance(attr1, MagicMock):
                continue
            elif isinstance(attr1, np.ndarray):
                assert np.array_equal(attr1, attr2)
            else:
                assert attr1 == attr2

        pass
