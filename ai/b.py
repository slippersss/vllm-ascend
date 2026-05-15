import inspect
from dataclasses import fields, is_dataclass
from types import MethodType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode, ModelConfig, VllmConfig
from vllm.v1.attention.backend import AttentionMetadataBuilder
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor, KVQuantMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor, CudaGraphManager, ModelCudaGraphManager
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.kv_connector import KVConnector
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend, AscendAttentionMetadataBuilder, AscendAttentionState, AscendMetadata
)
from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager
from vllm_ascend.worker.v2.block_table import AscendBlockTables
from vllm_ascend.worker.v2.input_batch import AscendInputBatch, AscendInputBuffers
from vllm_ascend.worker.v2.model_runner import NPUModelRunner
from vllm_ascend.worker.v2.model_states.default import AscendModelState
from vllm_ascend.worker.v2.states import AscendRequestState


class TestExecuteModelForGraph:
    max_num_reqs: int = 256
    max_num_toks: int = 32768
    cudagraph_capture_sizes: list[int] = [1, 2, 4, 8]

    def check_signature(self, target_function, parameters):
        parameter_set = set(inspect.signature(target_function).parameters.keys())
        return parameters.issubset(parameter_set)

    def check_attribute(self, target_class, fields, method="__init__"):
        if is_dataclass(target_class) and method == "__init__":
            field_set = set(target_class.__dataclass_fields__.keys())
            return fields.issubset(field_set)
        else:
            code = inspect.getsource(getattr(target_class, method))
            for field in fields:
                if f"self.{field}:" not in code and f"self.{field} =" not in code and f"self.{field}," not in code:
                    return False
            return True

    @pytest.fixture(scope="class", autouse=True)
    def prerequisite(self):
        assert hasattr(AscendBlockTables, "apply_staged_writes")
        assert self.check_signature(AscendBlockTables.apply_staged_writes, {"self"})

        assert hasattr(KVConnector, "no_forward")
        assert self.check_signature(KVConnector.no_forward, {"self", "scheduler_output"})
        assert hasattr(KVConnector, "pre_forward")
        assert self.check_signature(KVConnector.pre_forward, {"self", "scheduler_output"})
        assert hasattr(KVConnector, "post_forward")
        assert self.check_signature(KVConnector.post_forward, {"self", "scheduler_output"})

        assert self.check_attribute(AscendInputBuffers, {"seq_lens_cpu", "seq_lens_np", "query_start_loc"})
        assert self.check_attribute(InputBuffers, {"input_ids", "positions", "seq_lens"})

        assert self.check_attribute(AscendRequestState, {"num_computed_tokens_cpu"})
        assert self.check_attribute(
            RequestState,
            {
                "req_id_to_index",
                "next_prefill_tokens",
                "all_token_ids",
                "prefill_len",
                "last_sampled_tokens",
                "draft_tokens",
            },
        )
        assert hasattr(AscendRequestState, "any_prefills")
        assert self.check_signature(AscendRequestState.any_prefills, {"self", "idx_mapping_np"})

        assert self.check_attribute(
            AscendAttentionMetadataBuilder,
            {"decode_threshold", "speculative_config", "attn_mask_builder", "model_config"},
        )
        assert self.check_attribute(AttentionMetadataBuilder, {"kv_cache_spec", "device"})

        # assert hasattr(AttentionMaskBuilder, "get_attention_mask")  # not right
        # assert self.check_signature(AttentionMaskBuilder.get_attention_mask, {"self", "model_config"})  # not right

        assert self.check_attribute(ModelConfig, {"runner_type"}, method="__post_init__")

        assert self.check_attribute(NPUModelRunner, {"update_stream"})
        assert self.check_attribute(NPUModelRunner, {"model"}, method="load_model")
        assert self.check_attribute(
            GPUModelRunner,
            {
                "is_encoder_decoder",
                "dp_size",
                "dp_rank",
                "device",
                "max_num_reqs",
                "decode_query_len",
                "lora_config",
                "supports_mm_inputs",
                "is_first_pp_rank",
                "is_last_pp_rank",
                "use_aux_hidden_state_outputs",
                "kv_connector",
                "vllm_config",
                "input_buffers",
                "req_states",
                "speculative_config",
            },
        )
        assert self.check_attribute(
            GPUModelRunner,
            {"kv_cache_config", "attn_groups", "block_tables", "attn_backends"},
            method="initialize_kv_cache",
        )
        assert hasattr(NPUModelRunner, "finish_requests")
        assert self.check_signature(NPUModelRunner.finish_requests, {"self", "scheduler_output"})
        assert hasattr(NPUModelRunner, "free_states")
        assert self.check_signature(NPUModelRunner.free_states, {"self", "scheduler_output"})
        assert hasattr(NPUModelRunner, "add_requests")
        assert self.check_signature(NPUModelRunner.add_requests, {"self", "scheduler_output"})
        assert hasattr(NPUModelRunner, "update_requests")
        assert self.check_signature(NPUModelRunner.update_requests, {"self", "scheduler_output"})
        assert hasattr(NPUModelRunner, "_update_seq_lens_cpu")
        assert self.check_signature(NPUModelRunner._update_seq_lens_cpu, {"self", "scheduler_output", "req_ids"})
        assert hasattr(NPUModelRunner, "prepare_attn")
        assert self.check_signature(NPUModelRunner.prepare_attn, {"self", "input_batch"})

        assert self.check_attribute(ModelAclGraphManager, {"model_runner"})
        assert self.check_attribute(
            CudaGraphManager, {"_graphs_captured", "_candidates", "graphs", "device", "vllm_config"}
        )
        assert self.check_attribute(ModelCudaGraphManager, {"hidden_states", "use_aux_hidden_state_outputs"})

        assert self.check_attribute(DefaultModelState, {"max_model_len"})
        assert hasattr(DefaultModelState, "prepare_inputs")
        assert self.check_signature(DefaultModelState.prepare_inputs, {"self", "input_batch", "req_states"})

        import vllm_ascend.worker.v2.aclgraph_utils
        import vllm_ascend.worker.v2.model_runner
        import vllm.v1.worker.gpu.cudagraph_utils
        import vllm.v1.worker.gpu.model_runner
        assert self.check_signature(
            vllm_ascend.worker.v2.model_runner.prepare_prefill_inputs,
            {
                "input_ids",
                "next_prefill_tokens",
                "idx_mapping",
                "query_start_loc",
                "all_token_ids",
                "prefill_len",
                "num_computed_tokens",
            },
        )
        assert self.check_signature(
            vllm_ascend.worker.v2.model_runner.prepare_pos_seq_lens,
            {
                "idx_mapping",
                "query_start_loc",
                "num_computed_tokens",
                "pos",
                "seq_lens",
            },
        )
        assert self.check_signature(vllm_ascend.worker.v2.model_runner.update_cos_sin, {"positions"})
        assert self.check_signature(
            vllm.v1.worker.gpu.model_runner.build_slot_mappings_by_layer,
            {"slot_mappings", "kv_cache_config"},
        )
        assert self.check_signature(vllm.v1.worker.gpu.cudagraph_utils.get_offloader, set())
        print(inspect.signature(vllm_ascend.worker.v2.aclgraph_utils.set_forward_context).parameters.keys())
        assert self.check_signature(
            vllm_ascend.worker.v2.aclgraph_utils.set_forward_context,
            {"attn_metadata", "vllm_config", "num_tokens", "num_tokens_across_dp", "cudagraph_runtime_mode", "batch_descriptor", "slot_mapping"},
        )
        assert self.check_signature(
            vllm_ascend.worker.v2.aclgraph_utils.get_forward_context, set()
        )
        assert self.check_signature(
            vllm.v1.worker.gpu.model_runner.set_forward_context,
            {"attn_metadata", "vllm_config", "num_tokens", "num_tokens_across_dp", "cudagraph_runtime_mode", "batch_descriptor", "slot_mapping"},
        )
        assert self.check_signature(
            vllm_ascend.worker.v2.model_runner.build_attn_state,
            {
                "vllm_config",
                "seq_lens_np",
                "num_reqs",
                "num_scheduled_tokens",
                "num_valid_tokens",
            }
        )
        assert self.check_signature(
            vllm_ascend.worker.v2.model_runner.combine_sampled_and_draft_tokens,
            {
                "input_ids",
                "idx_mapping",
                "last_sampled_tokens",
                "query_start_loc",
                "seq_lens",
                "prefill_len",
                "draft_tokens",
                "cu_num_logits",
                "num_logits",
            }
        )

    def mock_block_tables(self):
        block_tables = MagicMock(spec=AscendBlockTables)
        block_tables.apply_staged_writes = MagicMock(return_value=None)
        return block_tables

    def mock_kv_connector(self):
        kv_connector = MagicMock(spec=KVConnector)
        kv_connector.no_forward = MagicMock(return_value=MagicMock())
        kv_connector.pre_forward = MagicMock(return_value=None)
        kv_connector.post_forward = MagicMock(return_value=MagicMock())
        return kv_connector

    def mock_vllm_config(self):
        vllm_config = MagicMock(spec=VllmConfig)
        return vllm_config

    def mock_input_buffers(self):
        input_buffers = MagicMock(spec=AscendInputBuffers)
        input_buffers.seq_lens_cpu = torch.zeros(self.max_num_reqs, dtype=torch.int32)
        input_buffers.seq_lens_np = input_buffers.seq_lens_cpu.numpy()
        input_buffers.query_start_loc = torch.zeros(self.max_num_reqs + 2, dtype=torch.int32)
        input_buffers.input_ids = torch.zeros(self.max_num_toks, dtype=torch.int32)
        input_buffers.positions = torch.zeros(self.max_num_toks, dtype=torch.int64)
        input_buffers.seq_lens = torch.zeros(self.max_num_reqs, dtype=torch.int32)
        return input_buffers

    def mock_req_states(self):
        req_states = MagicMock(spec=AscendRequestState)
        req_states.req_id_to_index = {}
        req_states.next_prefill_tokens = MagicMock()
        req_states.all_token_ids = MagicMock()
        req_states.prefill_len = MagicMock()
        req_states.num_computed_tokens = MagicMock()
        req_states.last_sampled_tokens = MagicMock()
        req_states.draft_tokens = MagicMock()
        return req_states

    @pytest.fixture
    def scheduler_output_one_prefill(self):
        new_request_data = NewRequestData(
            req_id="req_example",
            prompt_token_ids=list(range(262144, 262176)),
            mm_features=[],
            sampling_params=None,  # not important
            pooling_params=None,
            block_ids=([1],),
            num_computed_tokens=0,
            lora_request=None,
            prefill_token_ids=list(range(262144, 262176)),
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
    def scheduler_output_simple_mix(self):
        new_request_data = NewRequestData(
            req_id="req_prefill",
            prompt_token_ids=list(range(262144, 262148)),
            mm_features=[],
            sampling_params=None,  # not important
            pooling_params=None,
            block_ids=([2],),
            num_computed_tokens=0,
            lora_request=None,
            prefill_token_ids=list(range(262144, 262148)),
        )
        cached_request_data = CachedRequestData(
            req_ids=["req_decode"],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[None],
            num_computed_tokens=[4],
            num_output_tokens=[1],
        )

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[new_request_data],
            scheduled_cached_reqs=cached_request_data,
            num_scheduled_tokens={"req_prefill": 4, "req_decode": 1},
            total_num_scheduled_tokens=5,
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
    def kv_cache_config_attn_groups(self):
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
    def model_runner_basic_setting(self, request):
        mr = MagicMock(spec=NPUModelRunner)
        kv_cache_config, attn_groups, mock_metadata_builder = request.getfixturevalue("kv_cache_config_attn_groups")

        # fixed property
        mr.is_encoder_decoder = False
        mr.dp_size = 1
        mr.dp_rank = 0
        mr.device = torch.device("cpu")
        mr.max_num_reqs = self.max_num_reqs
        mr.decode_query_len = 1
        mr.lora_config = None
        mr.kv_cache_config = kv_cache_config
        mr.attn_groups = attn_groups
        mr.supports_mm_inputs = False
        mr.is_first_pp_rank = True
        mr.is_last_pp_rank = True
        mr.use_aux_hidden_state_outputs = False

        # non-target object
        mr.block_tables = self.mock_block_tables()
        mr.kv_connector = self.mock_kv_connector()
        mr.vllm_config = self.mock_vllm_config()
        mr.input_buffers = self.mock_input_buffers()
        mr.req_states = self.mock_req_states()

        # non-target function
        mr.finish_requests = MagicMock(return_value=None)
        mr.free_states = MagicMock(return_value=None)
        mr.add_requests = MagicMock(return_value=None)
        mr.update_requests = MagicMock(return_value=None)
        mr._update_seq_lens_cpu = MagicMock(return_value=None)
        mr.model = MagicMock(return_value=torch.zeros(10))

        # important object
        mr.cudagraph_manager = MagicMock(spec=ModelAclGraphManager)
        mr.cudagraph_manager._graphs_captured = True
        mr.cudagraph_manager._candidates = []
        for i in range(len(self.cudagraph_capture_sizes)):
            start = self.cudagraph_capture_sizes[i - 1] + 1 if i > 0 else 0
            end = self.cudagraph_capture_sizes[i] + 1
            for j in range(start, end):
                mr.cudagraph_manager._candidates.append(
                    [
                        BatchExecutionDescriptor(
                            cg_mode=CUDAGraphMode.FULL,
                            num_tokens=self.cudagraph_capture_sizes[i],
                            num_reqs=self.cudagraph_capture_sizes[i],
                        )
                    ]
                )
        mr.cudagraph_manager.dispatch = MethodType(ModelAclGraphManager.dispatch, mr.cudagraph_manager)  # target
        mr.cudagraph_manager.graphs = {}
        for i in range(len(self.cudagraph_capture_sizes)):
            batch_execution_description = BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.FULL,
                num_tokens=self.cudagraph_capture_sizes[i],
                num_reqs=self.cudagraph_capture_sizes[i],
            )
            graph = MagicMock()
            graph.replay = MagicMock(return_value=None)
            mr.cudagraph_manager.graphs[batch_execution_description] = graph
        mr.cudagraph_manager.is_last_pp_rank = True
        mr.cudagraph_manager.hidden_states = torch.zeros((self.cudagraph_capture_sizes[-1], 10))
        mr.cudagraph_manager.use_aux_hidden_state_outputs = False
        mr.cudagraph_manager.model_runner = mr
        mr.cudagraph_manager.model_runner.attn_backends = {MagicMock(): MagicMock()}
        mr.cudagraph_manager.model_runner.update_stream = MagicMock()
        mr.cudagraph_manager.model_runner.speculative_config = None
        mr.cudagraph_manager.device = torch.device("cpu")
        mr.cudagraph_manager.vllm_config = MagicMock()
        mr.cudagraph_manager.run_fullgraph = MethodType(
            ModelAclGraphManager.run_fullgraph, mr.cudagraph_manager
        )  # target

        mr.model_state = MagicMock(spec=AscendModelState)
        mr.model_state.max_model_len = self.max_num_toks
        mr.model_state.prepare_attn = MethodType(AscendModelState.prepare_attn, mr.model_state)  # target
        mr.model_state.prepare_inputs = MagicMock(return_value={})

        mock_metadata_builder.decode_threshold = 1
        mock_metadata_builder.kv_cache_spec = MagicMock()
        mock_metadata_builder.speculative_config = None
        mock_metadata_builder.attn_mask_builder = MagicMock()
        mock_metadata_builder.attn_mask_builder.get_attention_mask = MagicMock(
            return_value=torch.triu(torch.ones((2048, 2048), dtype=torch.int8), diagonal=1)
        )
        mock_metadata_builder.model_config = MagicMock()
        mock_metadata_builder.model_config.runner_type = "generate"
        mock_metadata_builder.device = torch.device("cpu")
        mock_metadata_builder.build = MethodType(AscendAttentionMetadataBuilder.build, mock_metadata_builder)  # target

        # important function
        mr.prepare_inputs = MethodType(NPUModelRunner.prepare_inputs, mr)  # target
        mr._pad_query_start_loc_for_fia = MethodType(NPUModelRunner._pad_query_start_loc_for_fia, mr)  # target
        mr.execute_model = MethodType(NPUModelRunner.execute_model, mr)  # target

        # specialized mock for specific scenario
        # vllm_ascend.worker.v2.model_runner.build_attn_state
        # self.req_states.req_id_to_index
        # self.req_states.any_prefills
        # vllm_ascend.worker.v2.model_runner.combine_sampled_and_draft_tokens
        # self.input_buffers.seq_lens
        # self.input_buffers.input_ids
        # self.input_buffers.positions
        # self.input_buffers.seq_lens_np
        # self.prepare_attn

        with (
            patch("vllm_ascend.worker.v2.model_runner.prepare_prefill_inputs"),
            patch("vllm_ascend.worker.v2.model_runner.prepare_pos_seq_lens"),
            patch("vllm_ascend.worker.v2.model_runner.update_cos_sin"),
            patch("vllm.v1.worker.gpu.model_runner.build_slot_mappings_by_layer"),
            patch("torch.Tensor.pin_memory", new=lambda self: self),
            patch("vllm.v1.worker.gpu.cudagraph_utils.get_offloader"),
            patch("vllm_ascend.worker.v2.aclgraph_utils.set_forward_context"),
            patch("vllm_ascend.worker.v2.aclgraph_utils.get_forward_context"),
            patch("vllm_ascend.worker.v2.aclgraph_utils.update_full_graph_params"),
            patch("vllm.v1.worker.gpu.model_runner.set_forward_context"),
        ):
            yield mr

    @pytest.fixture
    def model_runner_one_prefill(self, request):
        mr = request.getfixturevalue("model_runner_basic_setting")

        mr.req_states.req_id_to_index["req_example"] = 0
        mr.req_states.any_prefills = MagicMock(return_value=True)

        mr.input_buffers.seq_lens[0] = 32
        mr.input_buffers.input_ids[:32] = torch.arange(262144, 262176, dtype=torch.int32)
        mr.input_buffers.positions[:32] = torch.arange(0, 32, dtype=torch.int64)
        mr.input_buffers.seq_lens_cpu[0] = 32

        block_tables = (torch.zeros(666, dtype=torch.int32).reshape(1, -1),)
        block_tables[0][0][0] = 1
        slot_mappings = torch.arange(128, 160, dtype=torch.int64).reshape(1, -1)
        mr.prepare_attn = MagicMock(return_value=(block_tables, slot_mappings))

        with (
            patch("vllm_ascend.worker.v2.model_runner.build_attn_state") as mock_bas,
            patch("vllm_ascend.worker.v2.model_runner.combine_sampled_and_draft_tokens") as mock_csadt,
        ):
            mock_bas.return_value = AscendAttentionState.PrefillNoCache
            mock_csadt.return_value = torch.tensor([31], dtype=torch.int64)

            yield mr

    @pytest.fixture
    def model_runner_one_decode(self, request):
        mr = request.getfixturevalue("model_runner_basic_setting")

        mr.req_states.req_id_to_index["req_example"] = 0
        mr.req_states.any_prefills = MagicMock(return_value=False)

        mr.input_buffers.seq_lens[0] = 33
        mr.input_buffers.input_ids[:1] = torch.arange(262176, 262177, dtype=torch.int32)
        mr.input_buffers.positions[:1] = torch.arange(32, 33, dtype=torch.int64)
        mr.input_buffers.seq_lens_cpu[0] = 33

        block_tables = (torch.zeros(666, dtype=torch.int32).reshape(1, -1),)
        block_tables[0][0][0] = 1
        slot_mappings = torch.arange(160, 161, dtype=torch.int64).reshape(1, -1)
        mr.prepare_attn = MagicMock(return_value=(block_tables, slot_mappings))

        with (
            patch("vllm_ascend.worker.v2.model_runner.build_attn_state") as mock_bas,
            patch("vllm_ascend.worker.v2.model_runner.combine_sampled_and_draft_tokens") as mock_csadt,
        ):
            mock_bas.return_value = AscendAttentionState.DecodeOnly
            mock_csadt.return_value = torch.tensor([32], dtype=torch.int64)

            yield mr

    @pytest.fixture
    def model_runner_simple_mix(self, request):
        mr = request.getfixturevalue("model_runner_basic_setting")

        mr.req_states.req_id_to_index["req_decode"] = 0
        mr.req_states.req_id_to_index["req_prefill"] = 1
        mr.req_states.any_prefills = MagicMock(return_value=True)

        mr.input_buffers.seq_lens[0] = 5
        mr.input_buffers.seq_lens[1] = 4
        mr.input_buffers.input_ids[:1] = torch.arange(262148, 262149, dtype=torch.int32)
        mr.input_buffers.input_ids[1:5] = torch.arange(262144, 262148, dtype=torch.int32)
        mr.input_buffers.positions[:1] = torch.arange(4, 5, dtype=torch.int64)
        mr.input_buffers.positions[1:5] = torch.arange(0, 4, dtype=torch.int64)
        mr.input_buffers.seq_lens_cpu[0] = 5
        mr.input_buffers.seq_lens_cpu[1] = 4

        block_tables = (torch.zeros((2, 666), dtype=torch.int32),)
        block_tables[0][0][0] = 1
        block_tables[0][1][0] = 2
        slot_mappings = torch.cat(
            [
                torch.arange(132, 133, dtype=torch.int64),
                torch.arange(256, 260, dtype=torch.int64),
                -torch.ones(3, dtype=torch.int64),
            ]
        ).reshape(1, -1)
        mr.prepare_attn = MagicMock(return_value=(block_tables, slot_mappings))

        with (
            patch("vllm_ascend.worker.v2.model_runner.build_attn_state") as mock_bas,
            patch("vllm_ascend.worker.v2.model_runner.combine_sampled_and_draft_tokens") as mock_csadt,
        ):
            mock_bas.return_value = AscendAttentionState.ChunkedPrefill
            mock_csadt.return_value = torch.tensor([4, 3], dtype=torch.int64)

            yield mr

    @pytest.fixture
    def expected_result_one_prefill(self):
        block_tables = torch.zeros(666, dtype=torch.int32).reshape(1, -1)
        block_tables[0][0] = 1

        attn_metadata_one_layer = AscendMetadata(
            num_actual_tokens=32,
            num_decode_tokens=0,
            block_tables=block_tables,
            query_start_loc=torch.tensor([0, 32], dtype=torch.int32),
            seq_lens=torch.tensor([32], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([32], dtype=torch.int32),
            seq_lens_list=[32],
            max_query_len=32,
            actual_seq_lengths_q=[32],
            slot_mapping=torch.arange(128, 160, dtype=torch.int64),
            attn_mask=torch.triu(torch.ones((2048, 2048), dtype=torch.int8), diagonal=1),
            attn_state=AscendAttentionState.PrefillNoCache,
            num_prefills=1,
            num_decodes=0,
            causal=True,
            model_runner_type="generate",
            kvcomp_metadata=None,
        )

        seq_lens_np = np.zeros(256, dtype=np.int32)
        seq_lens_np[0] = 32

        input_batch = AscendInputBatch(
            req_ids=["req_example"],
            num_reqs=1,
            num_reqs_after_padding=1,
            idx_mapping=torch.tensor([0], dtype=torch.int32),
            idx_mapping_np=np.array([0], dtype=np.int32),
            expanded_idx_mapping=torch.tensor([0], dtype=torch.int32),
            expanded_local_pos=torch.tensor([0], dtype=torch.int32),
            num_scheduled_tokens=np.array([32], dtype=np.int32),
            num_tokens=32,
            num_tokens_after_padding=32,
            num_draft_tokens=0,
            query_start_loc=torch.tensor([0, 32], dtype=torch.int32),
            query_start_loc_np=np.array([0, 32], dtype=np.int32),
            seq_lens=torch.tensor([32], dtype=torch.int32),
            dcp_local_seq_lens=None,
            input_ids=torch.arange(262144, 262176, dtype=torch.int32),
            positions=torch.arange(0, 32, dtype=torch.int64),
            logits_indices=torch.tensor([31], dtype=torch.int64),
            cu_num_logits=torch.tensor([0, 1], dtype=torch.int32),
            cu_num_logits_np=np.array([0, 1], dtype=np.int32),
            has_structured_output_reqs=False,
            seq_lens_np=seq_lens_np,
            attn_state=AscendAttentionState.PrefillNoCache,
        )

        return attn_metadata_one_layer, input_batch

    @pytest.fixture
    def expected_result_one_decode(self):
        block_tables = torch.zeros(666, dtype=torch.int32).reshape(1, -1)
        block_tables[0][0] = 1

        attn_metadata_one_layer = AscendMetadata(
            num_actual_tokens=1,
            num_decode_tokens=1,
            block_tables=block_tables,
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            seq_lens=torch.tensor([33], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([33], dtype=torch.int32),
            seq_lens_list=[33],
            max_query_len=1,
            actual_seq_lengths_q=[1],
            slot_mapping=torch.arange(160, 161, dtype=torch.int64),
            attn_mask=torch.triu(torch.ones((2048, 2048), dtype=torch.int8), diagonal=1),
            attn_state=AscendAttentionState.DecodeOnly,
            num_prefills=0,
            num_decodes=1,
            causal=True,
            model_runner_type="generate",
            kvcomp_metadata=None,
        )

        seq_lens_np = np.zeros(256, dtype=np.int32)
        seq_lens_np[0] = 33

        input_batch = AscendInputBatch(
            req_ids=["req_example"],
            num_reqs=1,
            num_reqs_after_padding=1,
            idx_mapping=torch.tensor([0], dtype=torch.int32),
            idx_mapping_np=np.array([0], dtype=np.int32),
            expanded_idx_mapping=torch.tensor([0], dtype=torch.int32),
            expanded_local_pos=torch.tensor([0], dtype=torch.int32),
            num_scheduled_tokens=np.array([1], dtype=np.int32),
            num_tokens=1,
            num_tokens_after_padding=1,
            num_draft_tokens=0,
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            query_start_loc_np=np.array([0, 1], dtype=np.int32),
            seq_lens=torch.tensor([33], dtype=torch.int32),
            dcp_local_seq_lens=None,
            input_ids=torch.arange(262176, 262177, dtype=torch.int32),
            positions=torch.arange(32, 33, dtype=torch.int64),
            logits_indices=torch.tensor([32], dtype=torch.int64),
            cu_num_logits=torch.tensor([0, 1], dtype=torch.int32),
            cu_num_logits_np=np.array([0, 1], dtype=np.int32),
            has_structured_output_reqs=False,
            seq_lens_np=seq_lens_np,
            attn_state=AscendAttentionState.DecodeOnly,
        )

        return attn_metadata_one_layer, input_batch

    @pytest.fixture
    def expected_result_simple_mix(self):
        block_tables = torch.zeros((2, 666), dtype=torch.int32)
        block_tables[0][0] = 1
        block_tables[1][0] = 2

        attn_metadata_one_layer = AscendMetadata(
            num_actual_tokens=8,
            num_decode_tokens=1,
            block_tables=block_tables,
            query_start_loc=torch.tensor([0, 1, 5, 8], dtype=torch.int32),
            seq_lens=torch.tensor([5, 4, 0], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([5, 4, 0], dtype=torch.int32),
            seq_lens_list=[5, 4, 0],
            max_query_len=4,
            actual_seq_lengths_q=[1, 5, 8],
            slot_mapping=torch.cat(
                [
                    torch.arange(132, 133, dtype=torch.int64),
                    torch.arange(256, 260, dtype=torch.int64),
                    -torch.ones(3, dtype=torch.int64),
                ]
            ),
            attn_mask=torch.triu(torch.ones((2048, 2048), dtype=torch.int8), diagonal=1),
            attn_state=AscendAttentionState.ChunkedPrefill,
            num_prefills=2,
            num_decodes=1,
            causal=True,
            model_runner_type="generate",
            kvcomp_metadata=None,
        )

        seq_lens_np = np.zeros(256, dtype=np.int32)
        seq_lens_np[0] = 5
        seq_lens_np[1] = 4

        input_batch = AscendInputBatch(
            req_ids=["req_decode", "req_prefill"],
            num_reqs=2,
            num_reqs_after_padding=3,
            idx_mapping=torch.tensor([0, 1], dtype=torch.int32),
            idx_mapping_np=np.array([0, 1], dtype=np.int32),
            expanded_idx_mapping=torch.tensor([0, 1], dtype=torch.int32),
            expanded_local_pos=torch.tensor([0, 0], dtype=torch.int32),
            num_scheduled_tokens=np.array([1, 4], dtype=np.int32),
            num_tokens=5,
            num_tokens_after_padding=8,
            num_draft_tokens=0,
            query_start_loc=torch.tensor([0, 1, 5], dtype=torch.int32),  # not right
            query_start_loc_np=np.array([0, 1, 5, 8], dtype=np.int32),
            seq_lens=torch.tensor([5, 4], dtype=torch.int32),
            dcp_local_seq_lens=None,
            input_ids=torch.cat(
                [
                    torch.arange(262148, 262149, dtype=torch.int32),
                    torch.arange(262144, 262148, dtype=torch.int32),
                    torch.zeros(3, dtype=torch.int32),
                ]
            ),
            positions=torch.cat(
                [
                    torch.arange(4, 5, dtype=torch.int64),
                    torch.arange(0, 4, dtype=torch.int64),
                    torch.zeros(3, dtype=torch.int64),
                ]
            ),
            logits_indices=torch.tensor([4, 3], dtype=torch.int64),
            cu_num_logits=torch.tensor([0, 1, 2], dtype=torch.int32),
            cu_num_logits_np=np.array([0, 1, 2], dtype=np.int32),
            has_structured_output_reqs=False,
            seq_lens_np=seq_lens_np,
            attn_state=AscendAttentionState.ChunkedPrefill,
        )

        return attn_metadata_one_layer, input_batch

    @pytest.mark.parametrize("scenario", ["one_prefill", "one_decode", "simple_mix"])
    def test_execute_model_for_graph_with_scenario(self, request, scenario):
        scheduler_output = request.getfixturevalue(f"scheduler_output_{scenario}")
        model_runner = request.getfixturevalue(f"model_runner_{scenario}")

        model_runner.execute_model(scheduler_output=scheduler_output)

        attn_metadata = model_runner.execute_model_state.attn_metadata
        input_batch = model_runner.execute_model_state.input_batch
        expected_attn_metadata_one_layer, expected_input_batch = request.getfixturevalue(f"expected_result_{scenario}")

        assert len(attn_metadata) == 28, "num_layers of attn_metadata is not expected"
        for layer_index, layer_name in enumerate(attn_metadata):
            assert layer_name == f"model.layers.{layer_index}.self_attn.attn"
            for field in fields(expected_attn_metadata_one_layer):
                attr_output = getattr(attn_metadata[layer_name], field.name)
                attr_golden = getattr(expected_attn_metadata_one_layer, field.name)
                maybe_error_message = f"attn_metadata.{field.name} is not expected"
                if isinstance(attr_golden, torch.Tensor):
                    assert torch.equal(attr_output, attr_golden), maybe_error_message
                else:
                    assert attr_output == attr_golden, maybe_error_message

        for field in fields(expected_input_batch):
            attr_output = getattr(input_batch, field.name)
            attr_golden = getattr(expected_input_batch, field.name)
            maybe_error_message = f"input_batch.{field.name} is not expected"
            if isinstance(attr_golden, torch.Tensor):
                assert torch.equal(attr_output, attr_golden), maybe_error_message
            elif isinstance(attr_golden, np.ndarray):
                assert np.array_equal(attr_output, attr_golden), maybe_error_message
            else:
                assert attr_output == attr_golden, maybe_error_message

        assert hasattr(model_runner.model_state, "attn_metadata")

        import vllm_ascend.worker.v2.aclgraph_utils
        expected_call_count = 0 if scenario == "one_prefill" else 1
        assert vllm_ascend.worker.v2.aclgraph_utils.update_full_graph_params.call_count == expected_call_count
