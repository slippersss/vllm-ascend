
from unittest.mock import MagicMock, patch
import pytest
import torch

from types import MethodType


from vllm_ascend.worker.v2.model_runner import NPUModelRunner
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.core.sched.output import CachedRequestData
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor

from vllm_ascend.worker.v2.model_states.default import AscendModelState

from vllm_ascend.attention.attention_mask import AttentionMaskBuilder

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor, KVQuantMode
)
from vllm.v1.worker.gpu.kv_connector import KVConnector
from vllm.v1.worker.utils import AttentionGroup

from vllm_ascend.attention.attention_v1 import AscendAttentionBackend, AscendAttentionMetadataBuilder, AscendAttentionState
from vllm_ascend.attention.mla_v1 import AscendMLABackend, AscendMLAMetadataBuilder
from vllm_ascend.patch.platform.patch_kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager
from vllm_ascend.worker.v2.block_table import AscendBlockTables
from vllm_ascend.worker.v2.input_batch import AscendInputBuffers
from vllm_ascend.worker.v2.states import AscendRequestState

"""
I guess we need to restrict it to a paticular scenario
purely main model, no draft model, or lets say it's just the same as my offline script
and we just have some snapshots only
in this way we can just left so and some other things just right here and no need to take care of it
"""

class TestExecuteModelForGraph:
    max_num_reqs: int = 256
    max_num_toks: int = 32768
    cudagraph_capture_sizes: list[int] = [1, 2, 4, 8, 16, 32, 64]

    @pytest.fixture(scope="class", autouse=True)
    def prerequisite(self):
        pass

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
        input_buffers.seq_lens_cpu = torch.zeros(
            self.max_num_reqs,
            dtype=torch.int32
        )
        input_buffers.seq_lens_np = input_buffers.seq_lens_cpu.numpy()
        input_buffers.query_start_loc = torch.zeros(
            self.max_num_reqs + 2,
            dtype=torch.int32,
        )
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
        req_states.prefill_len = MagicMock()
        req_states.draft_tokens = MagicMock()
        return req_states

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
        kv_cache_config, attn_groups, mock_metadata_builder = request.getfixturevalue("kv_cache_config_attn_groups")

        mr = MagicMock(spec=NPUModelRunner)

        # fixed property
        mr.is_encoder_decoder = False
        mr.dp_size = 1
        mr.dp_rank = 0
        mr.device = torch.device("cpu")
        mr.max_num_reqs = self.max_num_reqs
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
        mr.model = MagicMock(return_value=torch.zeros(10, dtype=torch.bfloat16))

        # important object
        mr.cudagraph_manager = MagicMock(spec=ModelAclGraphManager)
        mr.cudagraph_manager._graphs_captured = True
        mr.cudagraph_manager._candidates = []
        for i in range(len(self.cudagraph_capture_sizes)):
            start = self.cudagraph_capture_sizes[i - 1] + 1 if i > 0 else 0
            end = self.cudagraph_capture_sizes[i] + 1
            for j in range(start, end):
                # FULL mode maybe
                mr.cudagraph_manager._candidates.append(
                    [
                        BatchExecutionDescriptor(
                            cg_mode=CUDAGraphMode.FULL,
                            num_tokens=self.cudagraph_capture_sizes[i],
                            num_reqs=self.cudagraph_capture_sizes[i],
                            uniform_token_count=1,
                        ),
                        BatchExecutionDescriptor(
                            cg_mode=CUDAGraphMode.FULL,
                            num_tokens=self.cudagraph_capture_sizes[i],
                            num_reqs=self.cudagraph_capture_sizes[i],
                        ),
                    ]
                )
        mr.cudagraph_manager.dispatch = MethodType(ModelAclGraphManager.dispatch, mr.cudagraph_manager)
        mr.cudagraph_manager.hidden_states = torch.zeros((64, 768), dtype=torch.bfloat16)
        mr.cudagraph_manager.use_aux_hidden_state_outputs = False
        mr.graphs = {}
        for i in range(len(self.cudagraph_capture_sizes)):
            batch_execution_description = BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.FULL,
                num_tokens=self.cudagraph_capture_sizes[i],
                num_reqs=self.cudagraph_capture_sizes[i],
                uniform_token_count=1,
            )
            graph = MagicMock(spec=torch.cuda.CUDAGraph)
            graph.replay() = MagicMock(return_value=None)
            mr.graphs[batch_execution_description] = graph

            batch_execution_description = BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.FULL,
                num_tokens=self.cudagraph_capture_sizes[i],
                num_reqs=self.cudagraph_capture_sizes[i],
            )
            graph = MagicMock(spec=torch.cuda.CUDAGraph)
            graph.replay() = MagicMock(return_value=None)
            mr.graphs[batch_execution_description] = graph
        mr.cudagraph_manager.run_fullgraph = MethodType(ModelAclGraphManager.run_fullgraph, mr.cudagraph_manager)

        mr.model_state = MagicMock(spec=AscendModelState)
        mr.model_state.prepare_attn = MethodType(AscendModelState.prepare_attn, mr.model_state)
        mr.model_state.max_model_len = self.max_num_toks
        mr.model_state.prepare_inputs = MagicMock(return_value={})

        mock_metadata_builder.decode_threshold = 1
        mock_metadata_builder.kv_cache_spec = None
        mock_metadata_builder.speculative_config = None
        mock_metadata_builder.model_config = MagicMock()
        mock_metadata_builder.model_config.hf_text_config = None
        mock_metadata_builder.model_config.runner_type = None
        mock_metadata_builder.device = torch.device("cpu")
        mock_metadata_builder.attn_mask_builder = AttentionMaskBuilder(torch.device("cpu"))
        mock_metadata_builder.build = MethodType(AscendAttentionMetadataBuilder.build, mock_metadata_builder)

        # important function
        mr.execute_model = MethodType(NPUModelRunner.execute_model, mr)
        mr.prepare_inputs = MethodType(NPUModelRunner.prepare_inputs, mr)

        with (
            patch("vllm_ascend.worker.v2.model_runner.prepare_prefill_inputs") as mock_0,
            patch("vllm_ascend.worker.v2.model_runner.prepare_pos_seq_lens") as mock_1,
            patch("vllm_ascend.worker.v2.model_runner.update_cos_sin") as mock_2,
            patch("vllm.v1.worker.gpu.model_runner.set_forward_context"),
            patch("vllm_ascend.worker_v2.aclgraph_utils.set_forward_context"),
            patch("vllm_ascend.worker_v2.aclgraph_utils.get_forward_context"),
            patch("vllm_ascend.worker_v2.aclgraph_utils.update_full_graph_params"),
            patch("vllm.v1.worker.gpu.cudagraph_utils.get_offloader"),
        ):
            mock_0.return_value = None
            mock_1.return_value = None
            mock_2.return_value = None

            yield mr

    @pytest.fixture
    def model_runner_one_prefill(self, request):
        mr = request.getfixturevalue("model_runner_basic_setting")

        mr.req_states.req_id_to_index["req_example"] = 0
        mr.req_states.any_prefills = MagicMock(return_value=True)

        ret_0 = (mr.block_tables.block_tables[0][:1],)
        ret_1 = torch.arange(128, 160).reshape(1, -1).to(torch.int32)
        mr.prepare_attn = MagicMock(return_value=(ret_0, ret_1))

        kv_cache_config, attn_groups, mock_builder = request.getfixturevalue("kv_cache_config_attn_groups")

        slot_mappings_by_layer = {}
        for layer_name in kv_cache_config.kv_cache_groups[0].layer_names:
            slot_mappings_by_layer[layer_name] = torch.arange(128, 160)

        # combine_sampled_and_draft_tokens
        with (
            patch("vllm_ascend.worker.v2.model_runner.build_attn_state") as mock_0,
            patch("vllm_ascend.worker.v2.model_runner.combine_sampled_and_draft_tokens") as mock_1,
            patch("torch.Tensor.pin_memory", new=lambda self: self),
            patch("vllm.v1.worker.gpu.model_runner.build_slot_mappings_by_layer") as mock_2,
        ):
            mock_0.return_value = AscendAttentionState.PrefillNoCache
            mock_1.return_value = torch.tensor([31], dtype=torch.int64)
            mock_2.return_value = {layer_name: torch.arange(128, 160) for layer_name in kv_cache_config.kv_cache_groups[0].layer_names}

            yield mr

    @pytest.fixture
    def model_runner_one_decode(self, request):
        mr = request.getfixturevalue("model_runner_basic_setting")

        mr.req_states.req_id_to_index["req_example"] = 0
        mr.req_states.any_prefills = MagicMock(return_value=False)

        # def mock_prepare_attn(self, input_batch):
        ret_0 = (mr.block_tables.block_tables[0][:1],)
        ret_1 = torch.arange(128, 160).reshape(1, -1).to(torch.int32)
        # return ret_0, ret_1
        mr.prepare_attn = MagicMock(return_value=(ret_0, ret_1))

        kv_cache_config, attn_groups, mock_builder = request.getfixturevalue("kv_cache_config_attn_groups")

        slot_mappings_by_layer = {}
        for layer_name in kv_cache_config.kv_cache_groups[0].layer_names:
            slot_mappings_by_layer[layer_name] = torch.arange(128, 160)

        # prepare_prefill_inputs
        # prepare_pos_seq_lens
        # combine_sampled_and_draft_tokens
        with (
            patch("vllm_ascend.worker.v2.model_runner.build_attn_state") as mock_0,
            patch("vllm_ascend.worker.v2.model_runner.combine_sampled_and_draft_tokens") as mock_1,
            patch("torch.Tensor.pin_memory", new=lambda self: self),
            patch("vllm_ascend.worker.v2.model_runner.prepare_prefill_inputs"),
            patch("vllm.v1.worker.gpu.model_runner.build_slot_mappings_by_layer") as mock_2,
        ):
            mock_0.return_value = AscendAttentionState.PrefillNoCache
            mock_1.return_value = torch.tensor([32], dtype=torch.int64)
            mock_2.return_value = slot_mappings_by_layer

            yield mr

    @pytest.mark.parametrize("scenario", ["one_prefill"])
    def test_execute_model_for_graph(self, request, scenario):
        so = request.getfixturevalue(f"scheduler_output_{scenario}")
        mr = request.getfixturevalue(f"model_runner_{scenario}")
        mr.execute_model(so)
        pass
