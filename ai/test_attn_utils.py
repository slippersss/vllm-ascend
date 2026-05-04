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


def get_param_names(sig):
    # sig = inspect.signature(fn)
    return [p.name for p in sig.parameters.values()]

class TestAttnUtils():
    @pytest.fixture(autouse=True, scope="class")
    def prerequisite(self):
        print(f"{'*' * 50} start checking")
        import vllm.v1.kv_cache_interface
        assert hasattr(vllm.v1.kv_cache_interface, "KVCacheConfig")
        assert "kv_cache_groups" in vllm.v1.kv_cache_interface.KVCacheConfig.__dataclass_fields__

        import vllm.v1.worker.utils
        assert hasattr(vllm.v1.worker.utils, "AttentionGroup")
        assert hasattr(vllm.v1.worker.utils.AttentionGroup, "get_metadata_builder")
        sig = inspect.signature(vllm.v1.worker.utils.AttentionGroup.get_metadata_builder)
        sig_name = get_param_names(sig)
        assert sig_name == ["self", "ubatch_id"]
        assert "layer_names" in vllm.v1.worker.utils.AttentionGroup.__dataclass_fields__

        import vllm.v1.attention.backend
        assert hasattr(vllm.v1.attention.backend, "AttentionMetadataBuilder")
        assert hasattr(vllm.v1.attention.backend.AttentionMetadataBuilder, "build")
        sig = inspect.signature(vllm.v1.attention.backend.AttentionMetadataBuilder.build)
        sig_name = get_param_names(sig)
        assert {"self", "common_prefix_len", "common_attn_metadata"} & set(sig_name) == {"self", "common_prefix_len", "common_attn_metadata"}

        import vllm_ascend.attention.utils
        assert hasattr(vllm_ascend.attention.utils, "AscendCommonAttentionMetadata")
        call_fields = {"query_start_loc", "query_start_loc_cpu", "seq_lens_cpu", "seq_lens", "num_reqs", "num_actual_tokens", "max_query_len", "block_table_tensor", "slot_mapping", "positions", "attn_state", "graph_pad_size", "num_input_tokens", "prefill_context_parallel_metadata", "max_seq_len"}
        assert call_fields & vllm_ascend.attention.utils.AscendCommonAttentionMetadata.__dataclass_fields__.keys() == call_fields


        # prerequisite @#####################################################################
        import vllm_ascend.attention.attention_mask
        assert hasattr(vllm_ascend.attention.attention_mask, "AttentionMaskBuilder")
        # This class is not correctly written (singlton), if you want to let the following work, you should rewrite
        # sig = inspect.signature(vllm_ascend.attention.attention_mask.AttentionMaskBuilder.__init__)
        # sig_name = get_param_names(sig)
        # assert sig_name == ["self", "device"]

        print(f"{'*' * 50} end checking")

        #######################################################################################

        import vllm.config
        assert hasattr(vllm.config, "VllmConfig")
        assert "model_config" in vllm.config.VllmConfig.__dataclass_fields__
        assert "speculative_config" in vllm.config.VllmConfig.__dataclass_fields__
        assert "method" in vllm.config.SpeculativeConfig.__dataclass_fields__
        assert "scheduler_config" in vllm.config.VllmConfig.__dataclass_fields__
        assert "enable_chunked_prefill" in vllm.config.SchedulerConfig.__dataclass_fields__

        assert hasattr(vllm.config, "ModelConfig")

        # we need to verify model_config can always have "runner_type", just simulate it
        model_config = vllm.config.ModelConfig("/home/data/Qwen3-0.6B")
        assert hasattr(model_config, "runner_type")
        # print(f"{model_config=}. {model_config.runner_type=}")

        # actually we need to verify vllm_config.kv_cache_config.kv_cache_groups[0].kv_cache_spec, but it's outdated now

        import vllm.v1.kv_cache_interface
        assert hasattr(vllm.v1.kv_cache_interface, "EncoderOnlyAttentionSpec")

        # The first branch will be throwed out

        import vllm_ascend.attention.attention_v1
        assert hasattr(vllm_ascend.attention.attention_v1, "AscendAttentionState")
        for name in ["PrefillNoCache", "PrefillCacheHit", "DecodeOnly", "ChunkedPrefill", "SpecDecoding"]:
            assert name in vllm_ascend.attention.attention_v1.AscendAttentionState.__members__
        # print(f"{vllm_ascend.attention.attention_v1.AscendAttentionState.__members__=}")
        # print(f"{vllm.config.VllmConfig.__dataclass_fields__=}")

        ####################################################################################
        import vllm.v1.kv_cache_interface
        assert hasattr(vllm.v1.kv_cache_interface, "KVCacheGroupSpec")
        some_fields = {"kv_cache_spec", "layer_names"}
        assert some_fields & vllm.v1.kv_cache_interface.KVCacheGroupSpec.__dataclass_fields__.keys() == some_fields
        assert hasattr(vllm.v1.kv_cache_interface, "UniformTypeKVCacheSpecs")


        assert hasattr(vllm.v1.kv_cache_interface, "KVCacheSpec")
        assert hasattr(vllm.v1.kv_cache_interface.KVCacheSpec, "page_size_bytes")

        assert hasattr(vllm.v1.kv_cache_interface, "MLAAttentionSpec")

        import vllm.config
        assert hasattr(vllm.config, "get_layers_from_vllm_config")
        sig = inspect.signature(vllm.config.get_layers_from_vllm_config)
        sig_name = get_param_names(sig)
        assert {"vllm_config", "layer_type", "layer_names"} & set(sig_name) == {"vllm_config", "layer_type", "layer_names"}

        assert hasattr(vllm.config, "get_current_vllm_config")
        # sig = inspect.signature(vllm.config.get_current_vllm_config)
        # sig_name = get_param_names(sig_name)
        ############ You actually need to make sure that every one got a default value so that you can call it directly


        import vllm.model_executor.layers.attention_layer_base
        assert hasattr(vllm.model_executor.layers.attention_layer_base, "AttentionLayerBase")

        import vllm.model_executor.layers.attention
        assert hasattr(vllm.model_executor.layers.attention, "MLAAttention")

        assert hasattr(vllm.config.VllmConfig, "kv_transfer_config")

        assert hasattr(vllm.config.KVTransferConfig, "is_kv_consumer")

        assert hasattr(vllm.v1.kv_cache_interface, "KVCacheTensor")
        assert "size" in vllm.v1.kv_cache_interface.KVCacheTensor.__dataclass_fields__
        assert "shared_by" in vllm.v1.kv_cache_interface.KVCacheTensor.__dataclass_fields__

        import vllm_ascend.quantization.utils
        assert hasattr(vllm_ascend.quantization.utils, "enable_fa_quant")
        sig = inspect.signature(vllm_ascend.quantization.utils.enable_fa_quant)
        
        sig_name = get_param_names(sig)
        assert "vllm_config" in sig_name

        ######################################################
        import vllm_ascend.utils
        assert hasattr(vllm_ascend.utils, "calc_split_factor")
        sig = inspect.signature(vllm_ascend.utils.calc_split_factor)
        
        sig_name = get_param_names(sig)
        assert "num_list" in sig_name

        import vllm.v1.attention.backend
        assert hasattr(vllm.v1.attention.backend, "AttentionBackend")
        assert hasattr(vllm.v1.attention.backend.AttentionBackend, "get_kv_cache_shape")
        sig = inspect.signature(vllm.v1.attention.backend.AttentionBackend.get_kv_cache_shape)
        sig_name = get_param_names(sig)
        expected_fields = {"num_blocks", "block_size", "num_kv_heads", "head_size", "cache_dtype_str"}
        assert expected_fields & set(sig_name) == expected_fields

        yield

    def print_mock(self, obj, prefix="", name=""):
        if isinstance(obj, MagicMock):
            print(f"{prefix} ({name}) {obj=}")
            dicts = obj.__dict__
            for key, value in dicts.items():
                if key == "method_calls" or key.startswith("_"):
                    continue
                self.print_mock(value, prefix + " " * 8, key)
        elif isinstance(obj, list):
            print(f"{prefix} ({name})")
            print(f"{prefix} begin a list {'*' * 20}")
            for item in obj:
                self.print_mock(item, prefix + " " * 8)
            print(f"{prefix} end a list {'*' * 20}")
        elif isinstance(obj, dict):
            print(f"{prefix} ({name})")
            print(f"{prefix} begin a dict {'*' * 20}")
            for item in obj.items():
                self.print_mock(item[1], prefix + " " * 8, item[0])
            print(f"{prefix} end a dict {'*' * 20}")
        else:
            print(f"{prefix} ({name}) {obj=}")

    @patch("vllm_ascend.worker.v2.attn_utils.AttentionMaskBuilder", new=MagicMock(return_value=MagicMock()))
    @patch("vllm_ascend.worker.v2.attn_utils._ATTENTION_MASK_BUILDER", new=None)
    def test_get_attn_mask_builder(self):
        import vllm_ascend.worker.v2.attn_utils
        MockAttentionMaskBuilder = vllm_ascend.worker.v2.attn_utils.AttentionMaskBuilder

        import vllm_ascend.worker.v2.attn_utils
        assert vllm_ascend.worker.v2.attn_utils._ATTENTION_MASK_BUILDER is None, "_ATTENTION_MASK_BUILDER should be None when init"

        device = MagicMock()
        init_builder = get_attn_mask_builder(device)
        assert init_builder is not None, "_ATTENTION_MASK_BUILDER init failed"
        assert init_builder is vllm_ascend.worker.v2.attn_utils._ATTENTION_MASK_BUILDER, "get wrong global builder"

        num_repetitions = 10
        for _ in range(num_repetitions):
            get_again_builder = get_attn_mask_builder(device)
            assert get_again_builder is not None, "_ATTENTION_MASK_BUILDER get failed"
            assert get_again_builder is init_builder, "Different _ATTENTION_MASK_BUILDER detected"

        assert MockAttentionMaskBuilder.call_count == 1, "_ATTENTION_MASK_BUILDER is init many times, more than 1"

        # print(f"{vllm_ascend.worker.v2.attn_utils._ATTENTION_MASK_BUILDER=}")

    @pytest.fixture
    def mocks(self):
        with (
            patch("vllm_ascend.worker.v2.attn_utils.AscendCommonAttentionMetadata") as mymock,
        ):
            mymock.retrun_value = MagicMock()
            yield

    @pytest.mark.parametrize("seq_lens_np", [None, np.full(123, 321, dtype=np.int32)])
    def test_build_attn_metadata(self, seq_lens_np, mocks):
        # better input params construction should be consider

        # mock_ss = self.mocks
        # prerequisite
        # with ExitStack() as stack:
            # stack.enter_context(patch("vllm_ascend.worker.v2.attn_utils."))
            # may be we need a random generator here for generating a batch
        backup_names = []
        backup_ret = {}

        attn_groups = [[MagicMock() for j in range(random.randint(0, 5))] for i in range(3)]
        backup_builder = []
        for i in range(3):
            temp = []
            for attn_group in attn_groups[i]:
                mock_builder = MagicMock()
                attn_group.get_metadata_builder = MagicMock(return_value=mock_builder)
                ret = MagicMock()
                mock_builder.build = MagicMock(return_value=ret)
                layer_names = ["111" + str(random.randint(0, 1000000)), "222" + str(random.randint(0, 1000000)), "333" + str(random.randint(0, 1000000))]
                attn_group.layer_names = layer_names
                backup_names.extend(layer_names)
                for layer_name in layer_names:
                    backup_ret[layer_name] = ret

                temp.append(mock_builder)

            backup_builder.append(temp)

        num_reqs = 256
        num_tokens = 32768

        query_start_loc_gpu = torch.tensor([0, 1, 5, 7, 9])  # act as if is gpu tensor, 1, 4, 2, 2
        query_start_loc_cpu = torch.tensor([0, 1, 5, 7, 9])

        max_query_len = 4
        seq_lens = torch.tensor([64, 12345, 672, 345])
        max_seq_len = 12345

        block_tables = torch.tensor([[45, 0, 0], [1, 0, 0], [301, 0, 0], [345, 0, 0]])

        slot_mappings = torch.zeros(num_tokens)

        kv_cache_config = MagicMock()
        kv_cache_config.kv_cache_groups = [MagicMock() for _ in range(3)]  # pretend that we have 10 groups

        dcp_local_seq_lens = None  # not use
        # seq_lens_np = np.full(num_reqs, max_seq_len, dtype=np.int32)  # can be none
        num_computed_tokens_cpu = None  # not use
        positions = torch.zeros(num_tokens)

        attn_state = MagicMock()

        graph_pad_size = random.randint(0, 100)
        num_input_tokens = 1000000
        prefill_context_parallel_metadata = MagicMock()  # can be anything, we don't care, we just pass it to the next

        # attempt to call build_attn_metadata
        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=query_start_loc_gpu,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            dcp_local_seq_lens=dcp_local_seq_lens,
            seq_lens_np=seq_lens_np,
            num_computed_tokens_cpu=num_computed_tokens_cpu,
            positions=positions,
            attn_state=attn_state,
            graph_pad_size=graph_pad_size,
            num_input_tokens=num_input_tokens,
            prefill_context_parallel_metadata=prefill_context_parallel_metadata,
        )

        # print(f"{attn_metadata=}")

        assert set(backup_names) == set(list(attn_metadata.keys()))

        assert set(list(attn_metadata.keys())) == set(list(backup_ret.keys()))

        for k, v in backup_ret.items():
            assert attn_metadata[k] is v

        import vllm_ascend.worker.v2.attn_utils
        assert vllm_ascend.worker.v2.attn_utils.AscendCommonAttentionMetadata.call_count == 3

        for i, group in enumerate(backup_builder):
            for j, builder in enumerate(group):
                assert builder.build.call_count == 1

        for i in range(3):
            for attn_group in attn_groups[i]:
                assert attn_group.get_metadata_builder.call_count == 1
                # assert

    def test_build_attn_state(self):
        # create more cases

        vllm_config = MagicMock()
        vllm_config.model_config = MagicMock()
        vllm_config.model_config.runner_type = "generate" # not pooling right now
        vllm_config.speculative_config = MagicMock()
        vllm_config.speculative_config.method = MagicMock()
        vllm_config.scheduler_config = MagicMock()
        vllm_config.scheduler_config.method = "mtp"
        seq_lens_np = np.array([43, 564, 32, 64, 643, 4235])
        num_reqs = 4
        num_scheduled_tokens = np.array([1, 2, 3, 4])
        num_valid_tokens = np.array([1, 1, 2, 1])

        attn_state = build_attn_state(
            vllm_config, seq_lens_np, num_reqs, num_scheduled_tokens, num_valid_tokens
        )
        # print(attn_state)

        assert attn_state == AscendAttentionState.ChunkedPrefill

        pass

    def test_get_layer_kv_cache_specs(self):
        import vllm.v1.kv_cache_interface
        kv_cache_config = MagicMock()
        kv_cache_config.kv_cache_groups = [MagicMock() for _ in range(3)]  # pretend that we have 10 groups
        all_layer_names = []
        pred_ret = {}
        for idx, group in enumerate(kv_cache_config.kv_cache_groups):
            layer_names = ["111" + str(random.randint(0, 1000000)), "222" + str(random.randint(0, 1000000)), "333" + str(random.randint(0, 1000000))]
            # print("?????????????", idx)
            if idx == 0:
                group.layer_names = layer_names
                group.kv_cache_spec = MagicMock(spec=vllm.v1.kv_cache_interface.UniformTypeKVCacheSpecs)

                group.kv_cache_spec.kv_cache_specs = {}
                for name in layer_names:
                    ret = MagicMock()
                    group.kv_cache_spec.kv_cache_specs[name] = ret

                    pred_ret[name] = ret

                # print("######################", group.kv_cache_specs)
            else:
                group.layer_names = layer_names
                ret = MagicMock()
                group.kv_cache_spec = ret

                for name in layer_names:
                    pred_ret[name] = ret

            all_layer_names.extend(layer_names)

        # print(f"{kv_cache_config.kv_cache_groups[0].kv_cache_specs=}")

        layer_kv_cache_spec = _get_layer_kv_cache_specs(kv_cache_config)
        # print(layer_kv_cache_spec)

        # assert set(list(layer_kv_cache_spec.keys())) == set(all_layer_names)
        # print(set(list(layer_kv_cache_spec.keys())))
        # print(set(all_layer_names))

        assert pred_ret == layer_kv_cache_spec

        pass

    @pytest.fixture
    def new_mock(self):
        import vllm.v1.kv_cache_interface
        import vllm.model_executor.layers.attention
        ret = MagicMock(spec=vllm.model_executor.layers.attention.MLAAttention)
        ret.kv_lora_rank = 123
        ret.qk_rope_head_dim = 321
        new_m = lambda a,b, layer_names: {name: ret for name in layer_names}
        with (
            patch("vllm_ascend.worker.v2.attn_utils.get_layers_from_vllm_config", new=new_m),
            patch("vllm_ascend.worker.v2.attn_utils.get_current_vllm_config", new=lambda: MagicMock()),
        ):
            
            yield

    def test_get_attention_kv_cache_dims(self, new_mock):
        # print(f"//////////////////////////////////////////////////////////////////////////////////{new_mock=}")
        import vllm.v1.kv_cache_interface
        layer_name = "eample.layer.name"
        kv_cache_spec = MagicMock(spec=vllm.v1.kv_cache_interface.MLAAttentionSpec)

        ret = _get_attention_kv_cache_dims(layer_name, kv_cache_spec)

        assert ret == (123, 321)

        # we left another branch to be resolved
        pass

    def test_align_memory(self):
        alignment = 2 * 1024 * 1024
        temp_tensor = torch.zeros(1536 + alignment, dtype=torch.int8)

        new_tensor = _align_memory(temp_tensor, alignment)

        assert new_tensor.element_size() == temp_tensor.element_size()
        assert (new_tensor.data_ptr() - temp_tensor.data_ptr()) % new_tensor.element_size() == 0
        assert new_tensor.data_ptr() % alignment == 0
        assert new_tensor.data_ptr() - temp_tensor.data_ptr() < alignment

        # print(new_tensor.data_ptr(), temp_tensor.data_ptr(), new_tensor.element_size())

        # ptr = (new_tensor.data_ptr() - temp_tensor.data_ptr()) // new_tensor.element_size() + (1536 + alignment) // alignment * alignment
        # after = temp_tensor.data_ptr() + alignment + 1536
        # assert ptr == after

        pass

    @pytest.mark.parametrize("mock_input", ["dsv2_lite_input_another", "qwen3_06b_input_another"])
    def test_allocate_kv_cache(self, mock_input, request):
        # mock_vllm_config.kv_transfer_config = None

        obj = request.getfixturevalue(mock_input)
        kv_cache_config, kv_cache_raw_tensors, attn_backends = obj

        my_result = _allocate_kv_cache(kv_cache_config, device=torch.device("cpu"))

        if mock_input == "dsv2_lite_input_another":
            all_layer_names = {f"model.layers.{i}.self_attn.attn" for i in range(27)}

            assert set(my_result.keys()) == set(all_layer_names)
            for key, value in my_result.items():
                assert len(value) == 2
                assert isinstance(value[0], torch.Tensor)
                assert isinstance(value[1], torch.Tensor)
                assert value[0].shape == (1024 * 128 * 576 * 1 * 2 * 512 // 576,)
                assert value[1].shape == (1024 * 128 * 576 * 1 * 2 * 64 // 576, )
        
        if mock_input == "qwen3_06b_input_another":
            all_layer_names = {f"model.layers.{i}.self_attn.attn" for i in range(28)}

            assert set(my_result.keys()) == set(all_layer_names)
            for key, value in my_result.items():
                assert len(value) == 2
                assert isinstance(value[0], torch.Tensor)
                assert isinstance(value[1], torch.Tensor)
                assert value[0].shape == (512 * 128 * 8 * 128 * 2,)
                assert value[1].shape == (512 * 128 * 8 * 128 * 2,)

    @pytest.fixture
    def mock_func_for_allocate_kv_cache(self):
        vllm_config = MagicMock()
        vllm_config.kv_transfer_config = None
        
        with patch("vllm_ascend.worker.v2.attn_utils.get_current_vllm_config", new=lambda: vllm_config), \
             patch("vllm_ascend.worker.v2.attn_utils._get_attention_kv_cache_dims", new=lambda a, b: (512, 64)):
            yield vllm_config

    @pytest.fixture
    def dsv2_lite_input_another(self):
        import vllm.v1.kv_cache_interface
        import vllm_ascend.attention.mla_v1
        num_layers = 27

        kv_cache_config = MagicMock()
        kv_cache_config.num_blocks = 1024   # you can set it in any way
        kv_cache_config.kv_cache_tensors = [MagicMock() for _ in range(num_layers)]
        kv_cache_config.kv_cache_groups = [MagicMock()] # maybe we can have multiple groups, this is just simulated dsv2 lite currently

        for i, item in enumerate(kv_cache_config.kv_cache_tensors):
            item.size = 1024 * 128 * 576 * 1 * 2  # num_blocks * block_size * num_heads * head_size * dtype in Bytes
            item.shared_by = [f"model.layers.{i}.self_attn.attn"]

        mock_layer_spec = {}

        kv_cache_group_spec = kv_cache_config.kv_cache_groups[0]

        kv_cache_group_spec.layer_names = []
        for i in range(27):
            kv_cache_group_spec.layer_names.append(f"model.layers.{i}.self_attn.attn")

        kv_cache_group_spec.kv_cache_spec = MagicMock(spec=vllm.v1.kv_cache_interface.MLAAttentionSpec)
        kv_cache_group_spec.kv_cache_spec.block_size = 128
        kv_cache_group_spec.kv_cache_spec.num_kv_heads = 1
        kv_cache_group_spec.kv_cache_spec.head_size = 576
        kv_cache_group_spec.kv_cache_spec.dtype = torch.bfloat16
        kv_cache_group_spec.kv_cache_spec.page_size_padded = None
        kv_cache_group_spec.kv_cache_spec.head_size_v = 576
        kv_cache_group_spec.kv_cache_spec.page_size_bytes = 128 * 576 * 2
        # kv_cache_group_spec.kv_cache_spec.sliding_window = None
        # kv_cache_group_spec.kv_cache_spec.attention_chunk_

        for i in range(num_layers):
            mock_layer_spec[f"model.layers.{i}.self_attn.attn"] = kv_cache_group_spec.kv_cache_spec

        kv_cache_raw_tensors = {}
        for i in range(num_layers):
            name = f"model.layers.{i}.self_attn.attn"
            kv_cache_raw_tensors[name] = (
                torch.zeros(1024 * 128 * 576 * 1 * 2 * 512 // 576, dtype=torch.int8), # 512 kv_lora_rank
                torch.zeros(1024 * 128 * 576 * 1 * 2 * 64 // 576, dtype=torch.int8), # 64 qk_rope_head_dim
            )

        attn_backends = {}
        for i in range(num_layers):
            name = f"model.layers.{i}.self_attn.attn"
            attn_backends[name] = MagicMock(spec=vllm_ascend.attention.mla_v1.AscendMLABackend)
            attn_backends[name].get_kv_cache_shape = lambda a, b, c, d, e: (a, b, c, d)


        vllm_config = MagicMock()
        vllm_config.kv_transfer_config = None

        with patch("vllm_ascend.worker.v2.attn_utils.get_current_vllm_config", new=lambda: vllm_config), \
             patch("vllm_ascend.worker.v2.attn_utils._get_attention_kv_cache_dims", new=lambda a, b: (512, 64)), \
             patch("vllm_ascend.worker.v2.attn_utils._get_layer_kv_cache_specs", new=lambda _: mock_layer_spec):
            yield kv_cache_config, kv_cache_raw_tensors, attn_backends

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
        # return kv_cache_config, kv_cache_raw_tensors, attn_backends

    @pytest.fixture
    def dsv2_lite_input(self):
        import vllm.v1.kv_cache_interface
        import vllm_ascend.attention.mla_v1
        num_layers = 27

        kv_cache_config = MagicMock()
        kv_cache_config.num_blocks = 1024   # you can set it in any way
        kv_cache_config.kv_cache_tensors = [MagicMock() for _ in range(num_layers)]
        kv_cache_config.kv_cache_groups = [MagicMock()] # maybe we can have multiple groups, this is just simulated dsv2 lite currently

        for i, item in enumerate(kv_cache_config.kv_cache_tensors):
            item.size = 1024 * 128 * 576 * 1 * 2  # num_blocks * block_size * num_heads * head_size * dtype in Bytes
            item.shared_by = [f"model.layers.{i}.self_attn.attn"]


        kv_cache_group_spec = kv_cache_config.kv_cache_groups[0]

        kv_cache_group_spec.layer_names = []
        for i in range(27):
            kv_cache_group_spec.layer_names.append(f"model.layers.{i}.self_attn.attn")

        kv_cache_group_spec.kv_cache_spec = MagicMock(spec=vllm.v1.kv_cache_interface.MLAAttentionSpec)
        kv_cache_group_spec.kv_cache_spec.block_size = 128
        kv_cache_group_spec.kv_cache_spec.num_kv_heads = 1
        kv_cache_group_spec.kv_cache_spec.head_size = 576
        kv_cache_group_spec.kv_cache_spec.dtype = torch.bfloat16
        kv_cache_group_spec.kv_cache_spec.page_size_padded = None
        kv_cache_group_spec.kv_cache_spec.head_size_v = 576
        kv_cache_group_spec.kv_cache_spec.page_size_bytes = 128 * 576 * 2
        # kv_cache_group_spec.kv_cache_spec.sliding_window = None
        # kv_cache_group_spec.kv_cache_spec.attention_chunk_

        kv_cache_raw_tensors = {}
        for i in range(num_layers):
            name = f"model.layers.{i}.self_attn.attn"
            kv_cache_raw_tensors[name] = (
                torch.zeros(1024 * 128 * 576 * 1 * 2 * 512 // 576, dtype=torch.int8), # 512 kv_lora_rank
                torch.zeros(1024 * 128 * 576 * 1 * 2 * 64 // 576, dtype=torch.int8), # 64 qk_rope_head_dim
            )

        attn_backends = {}
        for i in range(num_layers):
            name = f"model.layers.{i}.self_attn.attn"
            attn_backends[name] = MagicMock(spec=vllm_ascend.attention.mla_v1.AscendMLABackend)
            attn_backends[name].get_kv_cache_shape = lambda a, b, c, d, e: (a, b, c, d)

        # golden_result = {}
        # for i in range(num_layers):
        #     name = f"model.layers.{i}.self_attn.attn"
        #     golden_result[name] = (

        #     )

        vllm_config = MagicMock()
        vllm_config.kv_transfer_config = None
        
        with patch("vllm_ascend.worker.v2.attn_utils.get_current_vllm_config", new=lambda: vllm_config), \
             patch("vllm_ascend.worker.v2.attn_utils._get_attention_kv_cache_dims", new=lambda a, b: (512, 64)):
            yield kv_cache_config, kv_cache_raw_tensors, attn_backends

        # return kv_cache_config, kv_cache_raw_tensors, attn_backends

    @pytest.fixture
    def qwen3_06b_input(self):
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
             patch("vllm_ascend.worker.v2.attn_utils._get_attention_kv_cache_dims", new=lambda a, b: (128, 128)):
            yield kv_cache_config, kv_cache_raw_tensors, attn_backends
        # return kv_cache_config, kv_cache_raw_tensors, attn_backends

    @pytest.mark.parametrize("mock_input", ["dsv2_lite_input", "qwen3_06b_input"])
    def test_reshape_kv_cache(self, mock_input, request):
        obj = request.getfixturevalue(mock_input)
        kv_cache_config, kv_cache_raw_tensors, attn_backends = obj

        reshaped_kv_caches = _reshape_kv_cache(
            kv_cache_config,
            kv_cache_raw_tensors,
            attn_backends,
            None,
        )

        if mock_input == "dsv2_lite_input":
            all_layer_names = {f"model.layers.{i}.self_attn.attn" for i in range(27)}
            assert set(list(reshaped_kv_caches.keys())) == all_layer_names
            for key, value in reshaped_kv_caches.items():
                assert isinstance(value[0], torch.Tensor)
                assert isinstance(value[1], torch.Tensor)
                assert value[0].shape == (1024, 128, 1, 512)
                assert value[1].shape == (1024, 128, 1, 64)

        if mock_input == "qwen3_06b_input":
            all_layer_names = {f"model.layers.{i}.self_attn.attn" for i in range(28)}
            assert set(list(reshaped_kv_caches.keys())) == all_layer_names
            for key, value in reshaped_kv_caches.items():
                assert isinstance(value[0], torch.Tensor)
                assert isinstance(value[1], torch.Tensor)
                assert value[0].shape == (512, 128, 8, 128)
                assert value[1].shape == (512, 128, 8, 128)
