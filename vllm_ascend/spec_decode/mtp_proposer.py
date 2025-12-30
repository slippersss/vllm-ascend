import importlib
from typing import Union

import torch
import torch.nn as nn
from vllm.config import (CUDAGraphMode, get_layers_from_vllm_config,
                         set_current_vllm_config)
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.utils import \
    process_weights_after_loading
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm_ascend.spec_decode.eagle_proposer import EagleProposer

PADDING_SLOT_ID = -1

_MTP_MODELS = {
    "DeepseekV3ForCausalLM":
    ("vllm.model_executor.models.deepseek_mtp", "DeepSeekMTP"),
    "PanguUltraMoEForCausalLM":
    ("vllm.model_executor.models.openpangu_mtp", "OpenPanguMTP"),
    "DeepseekV32ForCausalLM":
    ("vllm.model_executor.models.deepseek_mtp", "DeepSeekMTP"),
    "Qwen3NextForCausalLM":
    ("vllm.model_executor.models.qwen3_next_mtp", "Qwen3NextMTP")
}


def _load_model(architecture):
    if architecture not in _MTP_MODELS:
        raise ValueError("Invalid architecture for mtp.")
    module_name, model_name = _MTP_MODELS[architecture]
    module = importlib.import_module(module_name)
    model = getattr(module, model_name)
    return model


class MtpProposer(EagleProposer):

    # TODO: Find out why ModelRunner does not this explicit typing?
    model: Union[nn.Module, ACLGraphWrapper]

    def load_model(self, model) -> None:
        loader = get_model_loader(self.vllm_config.load_config)

        target_attn_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config,
                                        AttentionLayerBase).keys())
        target_indexer_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config,
                                        DeepseekV32IndexerCache).keys())
        draft_model_config = \
            self.vllm_config.speculative_config.draft_model_config
        target_device = self.vllm_config.device_config.device

        with set_default_torch_dtype(
                draft_model_config.dtype), set_current_vllm_config(
                    self.vllm_config):
            self._init_mtp_model()
        draft_attn_layer_names = (get_layers_from_vllm_config(
            self.vllm_config, AttentionLayerBase).keys() -
                                  target_attn_layer_names)
        indexer_layers = get_layers_from_vllm_config(self.vllm_config,
                                                     DeepseekV32IndexerCache)
        draft_indexer_layer_names = indexer_layers.keys(
        ) - target_indexer_layer_names
        # NOTE: Currently we don't have specific attention backend and attention metadata
        # for deepseek v3.2 indexer, so we just exclude the indexer layers here.
        draft_attn_layer_names = draft_attn_layer_names - draft_indexer_layer_names

        assert len(draft_attn_layer_names) == 1
        self.attn_layer_names = list(draft_attn_layer_names)

        self.model.load_weights(
            loader.get_all_weights(
                self.vllm_config.speculative_config.draft_model_config,
                self.model))
        process_weights_after_loading(self.model, draft_model_config,
                                      target_device)

        if self.vllm_config.model_config.is_deepseek_mla:
            # check if mtp model use main model's embedding and LMhead
            main_model = model
            if get_pp_group().world_size == 1:
                # If pp>1, the weights of mtp and the main model's embedding are not on the same device.
                if torch.equal(self.model.model.embed_tokens.weight,
                               main_model.model.embed_tokens.weight):
                    self.model.model.embed_tokens = main_model.model.embed_tokens
            for _, layer_module in self.model.model.layers.items():
                if torch.equal(layer_module.shared_head.head.weight,
                               main_model.lm_head.weight):
                    layer_module.shared_head.head = main_model.lm_head

        if self.vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs(
        ):
            self.update_stream: torch.npu.Stream = torch.npu.Stream()
            self.model = ACLGraphWrapper(self.model,
                                         self.vllm_config,
                                         runtime_mode=CUDAGraphMode.FULL)

    def _init_mtp_model(self):
        architecture = self.vllm_config.model_config.architecture
        target_device = self.vllm_config.device_config.device
        model = _load_model(architecture)
        self.model = model(vllm_config=self.vllm_config).to(target_device)
