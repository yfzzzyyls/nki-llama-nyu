# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch LLaMA model for NXD inference."""
import copy
import gc
import logging
import math
from typing import List, Optional, Tuple, Type

import torch
from neuronx_distributed.parallel_layers import parallel_state  # noqa: E402
from neuronx_distributed.parallel_layers.layers import (  # noqa: E402; noqa: E402; noqa: E402; noqa: E402; noqa: E402
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers.mappings import (
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from neuronx_distributed.parallel_layers.utils import get_padding_length
from neuronx_distributed.quantization.quantization_config import QuantizationType, QuantizedDtype
from neuronx_distributed.quantization.quantization_layers import (  # noqa: E402; noqa: E402; noqa: E402; noqa: E402; noqa: E402
    QuantizedColumnParallel,
    QuantizedRowParallel,
)
from neuronxcc.nki._private_kernels.mlp import (
    mlp_fused_add_isa_kernel,
    mlp_isa_kernel,
    quant_mlp_fused_add_isa_kernel,
    quant_mlp_isa_kernel,
)
from neuronxcc.nki._private_kernels.rmsnorm import rmsnorm_quant_isa_kernel
from neuronxcc.starfish.penguin.targets.nki.private_api import vnc
from torch import nn, ones, Tensor
from torch_neuronx.xla_impl.ops import nki_jit
from transformers import LlamaForCausalLM
from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig  # noqa: E402
from neuronx_distributed_inference.models.model_base import (  # noqa: E402
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.gqa import (  # noqa: E402
    BaseGroupQueryAttention,
    GQA,
    GroupQueryAttention_O, 
    GroupQueryAttention_QKV,
)
from neuronx_distributed_inference.modules.attention.utils import (
    RotaryEmbedding,
    preprocess_quantized_linear_layer,
    transpose_parallel_linear_layer,
    apply_rotary_pos_emb,
    distributed_softmax,
    manual_softmax,
    move_heads_front,
    repeat_kv,
)

from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.flashdecode.utils import calculate_num_cores_per_group
from neuronx_distributed_inference.modules.lora_serving.lora_module import is_lora_module
from neuronx_distributed_inference.utils.distributed import get_tp_group

from torch_neuronx.xla_impl.ops import RmsNorm

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

# import time
from enum import Enum
class FlashAttentionStrategy(Enum):
    NONE = 0
    UNSHARDED_KERNEL = 1
    SHARDED_KERNEL = 2

SIMPLE_PROFILE = False

_LLAMA_MODULE_MAP = {}

### attention base class

import logging
import math
import warnings
from enum import Enum
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.distributed import ProcessGroup

# Try except for the compatibility with older compiler version
try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel  # noqa: E402
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel  # noqa: E402

import neuronx_distributed as nxd
import torch_xla.core.xla_model as xm
from neuronx_distributed.parallel_layers import utils  # noqa: E402
from neuronx_distributed.parallel_layers.layers import SPMDRank
from neuronx_distributed.parallel_layers.parallel_state import get_kv_shared_group
from neuronxcc.starfish.penguin.targets.nki.private_api import vnc
from torch_neuronx.xla_impl.ops import nki_jit  # noqa: E402
logger = logging.getLogger("Neuron")

_flash_fwd_call = nki_jit()(attention_isa_kernel)

class FlashAttentionStrategy(Enum):
    NONE = 0
    UNSHARDED_KERNEL = 1
    SHARDED_KERNEL = 2


class NeuronAttentionBase(nn.Module):
    """
    This base attention class implements the core Neuron related adaptation including
    1. replaces the q_proj, k_proj, v_proj with column parallel layer
    2. replaces the o_proj with row parallel layer
    3. update self.num_head to be self.num_head / tp_degree
    4. update self.num_key_value_heads to be self.num_key_value_heads / tp_degree
    5. update forward() method to adjust to changes from self.num_head
    """

    def __init__(self, tensor_model_parallel_group: Optional[ProcessGroup] = None):
        super().__init__()

        if tensor_model_parallel_group is not None:
            self.tensor_model_parallel_group = tensor_model_parallel_group
            self.rank_util = SPMDRank(world_size=self.tensor_model_parallel_group.size())
        elif nxd.parallel_layers.parallel_state.model_parallel_is_initialized():
            self.tensor_model_parallel_group = (
                nxd.parallel_layers.parallel_state.get_tensor_model_parallel_group()
            )
            self.rank_util = SPMDRank(world_size=self.tensor_model_parallel_group.size())
        else:
            # CPU flow doesn need rank_util and TP group now
            self.tensor_model_parallel_group = None
            self.rank_util = None

        self.is_causal = True
        self.num_key_value_groups = None
        self.num_key_value_heads = None
        self.num_heads = None
        self.rotary_emb = None
        self.o_proj = None
        self.qkv_proj = None
        self.bias = False
        self.k_layernorm = None
        self.q_layernorm = None
        self.qk_layernorm = False
        self.rms_norm_eps = None

        self.num_cores_per_group = 1
        self.flash_decoding_enabled = False
        self.sequence_parallel_enabled = False
        self.sequence_dimension = None
        self.rpl_reduce_dtype = None

        self.o_proj_layer_name = "o_proj"

    def init_gqa_properties(self):
        if (self.head_dim * self.num_attention_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_attention_heads})."
            )

        self.qkv_proj = GroupQueryAttention_QKV(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            tp_degree=self.tp_degree,
            dtype=self.torch_dtype,
            bias=self.bias,
            gather_output=False,
            fused_qkv=self.fused_qkv,
            clip_qkv=self.clip_qkv,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            sequence_dimension=self.sequence_dimension,
            tensor_model_parallel_group=self.tensor_model_parallel_group,
            rms_norm_eps=self.rms_norm_eps,
            qkv_kernel_enabled=self.neuron_config.qkv_kernel_enabled,
            logical_neuron_cores=self.neuron_config.logical_neuron_cores,
        )
        self.o_proj = GroupQueryAttention_O(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            tp_degree=self.tp_degree,
            dtype=self.torch_dtype,
            bias=self.bias,
            input_is_parallel=True,
            layer_name=self.o_proj_layer_name,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            sequence_dimension=self.sequence_dimension,
            tensor_model_parallel_group=self.tensor_model_parallel_group,
            rpl_reduce_dtype=self.rpl_reduce_dtype,
        )
        self.num_heads = utils.divide(self.qkv_proj.get_num_attention_heads(), self.tp_degree)
        self.num_key_value_heads = utils.divide(
            self.qkv_proj.get_num_key_value_heads(), self.tp_degree
        )
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        if self.qk_layernorm:
            self.q_layernorm = nn.LayerNorm(self.head_dim)
            self.k_layernorm = nn.LayerNorm(self.head_dim)
        self.attn_kernel_enabled = self.neuron_config.attn_kernel_enabled
        self.logical_neuron_cores = self.neuron_config.logical_neuron_cores

    def scaled_qk(self, Q, K, attention_mask):
        ## 可以操作
        ## [32,64] [64,32]
        bs, head, sequence, dimension = Q.size()
        _, head_k, sequence_k, dimension_k = K.size()
        # print("size:", sequence)
        # print("size2:", sequence_k)
        result = torch.zeros((bs, head, sequence, sequence_k), device = Q.device, dtype=Q.dtype)
        # print("Q.type:", Q.dtype)
        for i in range(bs):
            for j in range(head):
                temp_q = Q[i, j, :, :]
                temp_k = (K.transpose(2, 3))[i, j, :, :]
                # print("temp_q", temp_q.size())
                # print("temp_k", temp_k.size())
                temp = block_matmul(temp_q, temp_k)
                result[i, j, :, :] = temp
        QK = result / math.sqrt(self.head_dim)

        QK = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.head_dim)
        # print("Q", Q.size())
        # print("K", K.size())
        # print("QK:", QK.size())
        QK = torch.where(attention_mask, QK, torch.finfo(QK.dtype).min)
        return QK

    def prep_qkv_tensors(
        self,
        position_ids,
        hidden_states,
        past_key_value,
        adapter_ids=None,
        cos_cache=None,
        sin_cache=None,
        rmsnorm=None,
    ):
        """take care of the shape, layout, group query, custom position encoding, etc."""

        ## 可以操作
        ## [32, 2048][2048, 1024] // [32, 2048][2048, 256]
        Q = torch.matmul(hidden_states, self.qkv_proj.q_proj.weight.T)
        K = torch.matmul(hidden_states, self.qkv_proj.k_proj.weight.T)
        V = torch.matmul(hidden_states, self.qkv_proj.v_proj.weight.T)
        if self.qkv_proj.clip_qkv is not None:
            Q = Q.clamp(min=-self.clip_qkv, max=self.clip_qkv)
            K = K.clamp(min=-self.clip_qkv, max=self.clip_qkv)
            V = V.clamp(min=-self.clip_qkv, max=self.clip_qkv)

        # Q, K, V, cos_cache, sin_cache = self.prep_qkv_tensors(
        #     position_ids,
        #     hidden_states,
        #     past_key_value,
        #     adapter_ids=adapter_ids,
        #     cos_cache=cos_cache,
        #     sin_cache=sin_cache,
        #     rmsnorm=rmsnorm,
        # )

        # Divide hidden_dim across heads for MHA
        # Change layout: BSHD -> BHSD
        bsz, q_len, _ = hidden_states.size()
        if self.sequence_parallel_enabled:
            q_len *= self.tensor_model_parallel_group.size()

        Q = move_heads_front(
            Q, bsz, q_len, self.num_heads, self.head_dim, layernorm=self.q_layernorm
        )
        K = move_heads_front(
            K, bsz, q_len, self.num_key_value_heads, self.head_dim, layernorm=self.k_layernorm
        )
        V = move_heads_front(V, bsz, q_len, self.num_key_value_heads, self.head_dim, layernorm=None)

        # Rotate Q and K
        if self.rotary_emb is not None:
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(V, position_ids)

            Q, K = apply_rotary_pos_emb(Q, K, cos_cache, sin_cache)

        return Q, K, V, cos_cache, sin_cache

    def perform_prefill(self, Q, K, V, q_len, bsz, attention_mask) -> Tensor:
        """attention computation at prefilling (context encoding) phase"""
        K_active = repeat_kv(K, self.num_key_value_groups)
        V_active = repeat_kv(V, self.num_key_value_groups)

        flash_attn_strategy = self.get_flash_attention_strategy(q_len)
        logger.debug(f"Flash attention strategy: {flash_attn_strategy}")

        if flash_attn_strategy != FlashAttentionStrategy.NONE:
            logger.debug(f"ATTN kernel: logical_neuron_cores={self.logical_neuron_cores}")
            # if we are using left padding, then the bzs needs be 1 (otherwise we get wrong result
            # because flash attention does not use attention_mask). In practice, we use right
            # padding so this is unlikely to cause issues
            assert self.padding_side == "right" or bsz == 1

            # original shape of q, k, v is BHSD, and expected output is also BHSD.
            logger.debug(f"Using flash_fwd for Q.shape={Q.shape}")
            # make sure to cast inputs to torch_dtype (this is needed because the downcast to bf16
            # might happen after the kernel hlo creation step). Also convert shapes as expected by the kernel.

            # original Q shape: batch, num_heads, seqlen, d_head
            Q = (
                Q.permute(0, 1, 3, 2)  # after permute: batch, num_heads, d_head, seqlen
                .reshape((bsz * self.num_heads, self.head_dim, q_len))
                .to(self.torch_dtype)
            )
            Q = Q / math.sqrt(self.head_dim)
            K_active = (
                K_active.permute(0, 1, 3, 2)
                .reshape((bsz * self.num_heads, self.head_dim, q_len))
                .to(self.torch_dtype)
            )
            V_active = V_active.reshape((bsz * self.num_heads, q_len, self.head_dim)).to(
                self.torch_dtype
            )
            # shape: (B*H)DS
            attn_output = torch.zeros(
                bsz * self.num_heads, self.head_dim, q_len, dtype=Q.dtype, device=Q.device
            )

            logger.debug("Input parameter shapes")
            logger.debug(f"Q input shape {Q.shape}")
            logger.debug(f"K input shape {K_active.shape}")
            logger.debug(f"V input shape {V_active.shape}")
            logger.debug(f"Attn output shape {attn_output.shape}")

            if flash_attn_strategy == FlashAttentionStrategy.SHARDED_KERNEL:
                grid = (vnc(self.logical_neuron_cores),)

                _flash_fwd_call[grid](
                    Q,
                    K_active,
                    V_active,
                    1.0,
                    attn_output,
                    kernel_name="CausalAttentionMMSoftmaxMMWithoutSwap",
                )
            elif flash_attn_strategy == FlashAttentionStrategy.UNSHARDED_KERNEL:
                _flash_fwd_call(
                    Q,
                    K_active,
                    V_active,
                    1.0,
                    attn_output,
                    kernel_name="CausalAttentionMMSoftmaxMMWithoutSwap",
                )
            else:
                raise ValueError(f"Invalid flash attention strategy: {flash_attn_strategy}")

            # shape: BHDS
            attn_output = attn_output.reshape((bsz, self.num_heads, self.head_dim, q_len))
            logger.debug(f"Attn output after reshape {attn_output.shape}")
        else:
            logger.debug("ATTN: native compiler")
            logger.debug(f"Not using flash_fwd for Q.shape={Q.shape}")
            active_scores = self.scaled_qk(Q, K_active, attention_mask)
            active_scores = nn.functional.softmax(active_scores, dim=-1, dtype=torch.float32).to(
                Q.dtype
            )
            
            # 可以操作
            # print("active_scores:", active_scores.size())
            # print("V_active:", V_active.size())
            ## [1,16,32,32] [1,16,32,64]

            ###(有格式问题)
            # active_scores = active_scores.to(Q.dtype)
            # V_active = V_active.to(Q.dtype)
            # bs, head, sequence, dimension = active_scores.size()
            # _, head_1, sequence_1, dimension_1 = V_active.size()
            # # print("active_scores.type:", active_scores.dtype)
            # # print("V_active.type:", V_active.dtype)
            # # print("size:", sequence)
            # # print("size2:", sequence_k)
            # result = torch.zeros((bs, head, sequence, dimension_1), device = active_scores.device, dtype=active_scores.dtype)
            # for i in range(bs):
            #     for j in range(head):
            #         temp_q = active_scores[i, j, :, :]
            #         temp_k = V_active[i, j, :, :]
            #         temp = block_matmul(temp_q, temp_k)
            #         result[i, j, :, :] = temp
            # attn_output = result
            ###

            attn_output = torch.matmul(active_scores, V_active)
        return attn_output, flash_attn_strategy

    def get_flash_attention_strategy(self, q_len) -> FlashAttentionStrategy:
        """
        Gets the flash attention strategy.

        For LNC1, use the unsharded kernel if sequence length is at least 4096 to get the best performance.
        The unsharded kernel requires a sequence length of at least 512.

        For LNC2, use the sharded kernel if sequence length is divisible by 1024. Otherwise, use no
        kernel, because the unsharded kernel has worse performance than no kernel.
        The sharded kernel requires a sequence length of at least 1024.

        These constraints may change later.

        TODO: Throw an exception instead of disabling flash attention if explicitly enabled but not eligible.
              This must consider bucketing to avoid throwing an exception for smaller buckets.
        """
        if int(self.logical_neuron_cores) > 1:
            if q_len < 1024:
                return FlashAttentionStrategy.NONE

            if q_len % 1024 == 0:
                return FlashAttentionStrategy.SHARDED_KERNEL
            else:
                warnings.warn(
                    "Flash attention disabled. LNC2 requires seq_len % 1024 for flash attn to be performant"
                )
                return FlashAttentionStrategy.NONE

        # If seq_len is at least 4096, enable flash attn automatically to improve performance.
        if q_len >= 4096:
            return FlashAttentionStrategy.UNSHARDED_KERNEL

        # At lower seq lens, enable only if explicitly enabled.
        if self.attn_kernel_enabled and q_len >= 512:
            return FlashAttentionStrategy.UNSHARDED_KERNEL

        return FlashAttentionStrategy.NONE

    def compute_for_flash_decoding(
        self, Q, K, V, past_key_value, attention_mask, active_mask
    ) -> Tensor:
        # TODO: refactor/decompose this to reduce duplication with compute_for_token_gen
        # active attention
        n_repeat = Q.shape[1]
        K_active = repeat_kv(K, n_repeat)
        V_active = repeat_kv(V, n_repeat)

        # 可以操作(bujingguo)
        ## [32,64] [64,32]
        active_scores = (torch.matmul(Q, K_active.transpose(2, 3)) / math.sqrt(self.head_dim)).to(
            torch.float32
        )
        active_scores = torch.where(
            active_mask, active_scores, torch.finfo(active_scores.dtype).min
        )

        # prior attention
        K_prior = repeat_kv(past_key_value[0], n_repeat)
        V_prior = repeat_kv(past_key_value[1], n_repeat)

        # 可以操作(bujinguo)

        prior_scores = torch.matmul(Q, K_prior.transpose(2, 3)) / math.sqrt(self.head_dim)
        prior_scores = torch.where(
            attention_mask, prior_scores, torch.finfo(prior_scores.dtype).min
        )
        prior_scores = prior_scores.to(torch.float32)

        # attention scores
        softmax_prior, softmax_active = distributed_softmax(prior_scores, active_scores)
        softmax_prior, softmax_active = softmax_prior.to(Q.dtype), softmax_active.to(Q.dtype)

        ## 可以操作(bujingguo)

        # print("softmax_prior", softmax_prior.size())
        # print("V_prior:", V_prior.size())
        # print("soft_active:", softmax_active.size())
        # print("V_active:", V_active.size())
        attn_prior = torch.matmul(softmax_prior, V_prior)
        attn_active = torch.matmul(softmax_active, V_active)
        attn_output = attn_prior + attn_active

        return attn_output

    def compute_for_token_gen(
        self, Q, K, V, position_ids, past_key_value, attention_mask, active_mask
    ) -> Tensor:
        """attention computation at token generation phase"""
        is_speculation = position_ids.shape[-1] > 1

        # Attention computation: softmax((Q.K/√dkv) + mask).V
        # i. prior (cached) KV
        K_prior = past_key_value[0]
        V_prior = past_key_value[1]
        K_prior = repeat_kv(K_prior, self.num_key_value_groups)
        V_prior = repeat_kv(V_prior, self.num_key_value_groups)

        ## 可以操作
        
        prior_scores = torch.matmul(Q, K_prior.transpose(2, 3)) / math.sqrt(self.head_dim)
        prior_scores = torch.where(
            attention_mask, prior_scores, torch.finfo(prior_scores.dtype).min
        )
        prior_scores = prior_scores.to(torch.float32)

        # ii. active (current/new) KV
        K_active = repeat_kv(K, self.num_key_value_groups)
        V_active = repeat_kv(V, self.num_key_value_groups)

        ## 可以操作

        active_scores = torch.matmul(Q, K_active.transpose(2, 3)) / math.sqrt(self.head_dim)
        if is_speculation:
            active_scores = torch.where(
                active_mask, active_scores, torch.finfo(active_scores.dtype).min
            )
        active_scores = active_scores.to(torch.float32)

        # iii. attention scores
        softmax_prior, softmax_active = manual_softmax(prior_scores, active_scores, is_speculation)
        softmax_prior, softmax_active = softmax_prior.to(Q.dtype), softmax_active.to(Q.dtype)

        ## 可以操作
        #[1,16,1,64][1,16,64,64]
        #[1,16,1,1][1,16,1,64]
        # print("softmax_prior", softmax_prior.size())
        # print("V_prior:", V_prior.size())
        # print("soft_active:", softmax_active.size())
        # print("V_active:", V_active.size())

        ###
        # softmax_prior = softmax_prior.to(Q.dtype)
        # V_prior = V_prior.to(Q.dtype)
        # bs, head, sequence, dimension = softmax_prior.size()
        # _, head_1, sequence_1, dimension_1 = V_prior.size()
        # result = torch.zeros((bs, head, sequence, dimension_1), device = softmax_prior.device, dtype=softmax_prior.dtype)
        # for i in range(bs):
        #     for j in range(head):
        #         temp_q = softmax_prior[i, j, :, :]
        #         temp_k = V_prior[i, j, :, :]
        #         temp = block_matmul(temp_q, temp_k)
        #         result[i, j, :, :] = temp
        # attn_prior = result
        ###

        ###
        # softmax_active = softmax_active.to(Q.dtype)
        # V_active = V_active.to(Q.dtype)
        # bs, head, sequence, dimension = softmax_active.size()
        # _, head_1, sequence_1, dimension_1 = V_active.size()
        # result = torch.zeros((bs, head, sequence, dimension_1), device = softmax_active.device, dtype=softmax_active.dtype)
        # for i in range(bs):
        #     for j in range(head):
        #         temp_q = softmax_active[i, j, :, :]
        #         temp_k = V_active[i, j, :, :]
        #         temp = block_matmul(temp_q, temp_k)
        #         result[i, j, :, :] = temp
        # attn_active = result
        ###

        attn_prior = torch.matmul(softmax_prior, V_prior)
        attn_active = torch.matmul(softmax_active, V_active)
        attn_output = attn_prior + attn_active

        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        active_mask: Optional[torch.LongTensor] = None,
        adapter_ids=None,
        cos_cache: Optional[torch.Tensor] = None,
        sin_cache: Optional[torch.Tensor] = None,
        rmsnorm=None,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """Implements each layer's forward pass for the attention block."""
        bsz, q_len, _ = hidden_states.size()

        Q, K, V, cos_cache, sin_cache = self.prep_qkv_tensors(
            position_ids,
            hidden_states,
            past_key_value,
            adapter_ids=adapter_ids,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            rmsnorm=rmsnorm,
        )

        if past_key_value is None:
            attn_output, _ = self.perform_prefill(
                Q, K, V, q_len, bsz, attention_mask
            )
        else:
            attn_output = self.compute_for_token_gen(
                Q, K, V, position_ids, past_key_value, attention_mask, active_mask
            )


        attn_output = attn_output.transpose(1, 2).contiguous()

        # merge multi head hidden
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        # Z = Z.Wo
        attn_output = self.o_proj(attn_output, adapter_ids=adapter_ids)

        past_key_value: Tuple[Tensor, Tensor] = (K, V)

        return attn_output, past_key_value, cos_cache, sin_cache

## attention base class

@nki.jit
def nki_matmul_fully_optimized_spmd_(
    lhsT,
    rhs,
    TILES_IN_BLOCK_M=1,
    TILES_IN_BLOCK_N=16,
    TILES_IN_BLOCK_K=8,
    # Number of parallel “chunks” (SPMD workers) we want along the M dimension:
    spmd_m=2
):
    """
    SPMD variant of your matmul kernel, distributing the M blocks among `spmd_m` workers.
    Each worker runs the same code, but only processes a slice of the M dimension.
    """

    # 1) Validate shapes
    K, M = lhsT.shape
    K_, N = rhs.shape
    assert K == K_, "lhsT and rhs must have the same contraction dimension"

    # 2) Allocate the final result in shared HBM
    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    # 3) Define tile sizes (as before)
    TILE_M = nl.tile_size.gemm_stationary_fmax  # usually 128
    TILE_K = nl.tile_size.pmax                  # usually 128
    TILE_N = nl.tile_size.gemm_moving_fmax      # usually 512

    # 4) Compute the block sizes
    BLOCK_M = TILE_M * TILES_IN_BLOCK_M
    BLOCK_N = TILE_N * TILES_IN_BLOCK_N
    BLOCK_K = TILE_K * TILES_IN_BLOCK_K

    # 5) Ensure the global shapes are multiples of these blocks
    assert M % BLOCK_M == 0, "M not divisible by BLOCK_M"
    assert N % BLOCK_N == 0, "N not divisible by BLOCK_N"
    assert K % BLOCK_K == 0, "K not divisible by BLOCK_K"

    # 6) Count how many blocks along each dimension
    NUM_BLOCK_M = M // BLOCK_M  # total blocks in M dimension
    NUM_BLOCK_N = N // BLOCK_N  # total blocks in N dimension
    NUM_BLOCK_K = K // BLOCK_K  # total blocks in K dimension

    # ------------------------------------------------------------------
    # 7) Figure out which sub-range of M-blocks *this SPMD worker* handles
    #    We assume we want to distribute M across spmd_m cores.
    #    So we chunk the M-block range [0..NUM_BLOCK_M) into spmd_m pieces.
    # ------------------------------------------------------------------
    # This kernel’s "worker id" along dimension 0:
    my_m_id = nl.program_id(0)  # 0,1,... up to (spmd_m-1)

    # Blocks per worker in M dimension:
    blocks_per_worker = NUM_BLOCK_M // spmd_m
    assert (NUM_BLOCK_M % spmd_m) == 0, \
        "For simplicity, we assume NUM_BLOCK_M is divisible by spmd_m"

    start_m = my_m_id * blocks_per_worker
    end_m   = (my_m_id + 1) * blocks_per_worker

    # 8) Outer loop on N dimension
    for n in nl.affine_range(NUM_BLOCK_N):

        # We’ll store partial results in SBUF as you already do
        result_tiles = nl.zeros(
            (blocks_per_worker, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N,
             nl.par_dim(TILE_M), TILE_N),
            dtype=lhsT.dtype,
            buffer=nl.sbuf
        )

        # 9) Loop over K dimension
        for k_blk in nl.sequential_range(NUM_BLOCK_K):
            # -- Load tiles from RHS
            i_rhs = nl.mgrid[0:TILE_K, 0:BLOCK_N]
            rhs_tiles = nl.ndarray(
                (TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_N),
                dtype=rhs.dtype,
                buffer=nl.sbuf
            )

            for bk_r in nl.affine_range(TILES_IN_BLOCK_K):
                rhs_tiles[bk_r, i_rhs.p, i_rhs.x] = nl.load(
                    rhs[(TILES_IN_BLOCK_K * k_blk + bk_r) * TILE_K + i_rhs.p,
                        BLOCK_N * n + i_rhs.x]
                )

            # 10) Now loop over *our subset* of M blocks
            for local_m_index in nl.affine_range(blocks_per_worker):
                # The global M block index this worker is handling
                m = start_m + local_m_index

                # -- Load tiles from LHS^T
                i_lhsT = nl.mgrid[0:TILE_K, 0:BLOCK_M]
                lhsT_tiles = nl.ndarray(
                    (TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_M),
                    dtype=lhsT.dtype,
                    buffer=nl.sbuf
                )
                for bk_l in nl.affine_range(TILES_IN_BLOCK_K):
                    lhsT_tiles[bk_l, i_lhsT.p, i_lhsT.x] = nl.load(
                        lhsT[(TILES_IN_BLOCK_K * k_blk + bk_l) * TILE_K + i_lhsT.p,
                             BLOCK_M * m + i_lhsT.x]
                    )

                # -- Perform the partial matmul
                i_lhsT_mm = nl.mgrid[0:TILE_K, 0:TILE_M]
                i_rhs_mm  = nl.mgrid[0:TILE_K, 0:TILE_N]
                i_res_mm  = nl.mgrid[0:TILE_M, 0:TILE_N]
                for bn in nl.affine_range(TILES_IN_BLOCK_N):
                    for bm in nl.affine_range(TILES_IN_BLOCK_M):
                        res_tile = nl.zeros((TILE_M, TILE_N),
                                            dtype=nl.float32, buffer=nl.psum)
                        for bk in nl.affine_range(TILES_IN_BLOCK_K):
                            res_tile[...] += nisa.nc_matmul(
                                lhsT_tiles[bk, i_lhsT_mm.p, bm*TILE_M + i_lhsT_mm.x],
                                rhs_tiles[bk, i_rhs_mm.p, bn*TILE_N + i_rhs_mm.x]
                            )
                        # Accumulate partial sums into SBUF tile
                        result_tiles[local_m_index, bm, bn,
                                     i_res_mm.p, i_res_mm.x] += \
                            res_tile[i_res_mm.p, i_res_mm.x]

        # 11) Write final tiles from SBUF → shared HBM
        for local_m_index in nl.affine_range(blocks_per_worker):
            # The global M block index
            m = start_m + local_m_index
            for bm in nl.affine_range(TILES_IN_BLOCK_M):
                i_res        = nl.mgrid[0:TILE_K, 0:TILE_N]
                i_res_packed = nl.mgrid[0:TILE_K, 0:BLOCK_N]
                result_packed = nl.ndarray((TILE_K, BLOCK_N),
                                           dtype=result_tiles.dtype,
                                           buffer=nl.sbuf)

                # coalesce result tiles for better DMA performance
                for bn in nl.affine_range(TILES_IN_BLOCK_N):
                    result_packed[i_res.p, bn*TILE_N + i_res.x] = nl.copy(
                        result_tiles[local_m_index, bm, bn, i_res.p, i_res.x]
                    )
                # Now store from SBUF to final result in HBM
                nl.store(
                    result[(TILES_IN_BLOCK_M*m + bm)*TILE_K + i_res_packed.p,
                           BLOCK_N*n + i_res_packed.x],
                    value=result_packed[i_res_packed.p, i_res_packed.x]
                )

    return result

@nki.jit
def nki_matmul_fully_optimized_spmd_proj_fused(
    lhsT,         # shape (K, M)
    rhs,          # shape (K, N)
    TILES_IN_BLOCK_M=1,
    TILES_IN_BLOCK_N=16,
    TILES_IN_BLOCK_K=16,
    spmd_m=2      # 2 SPMD workers along M dimension
):
    """
    Example SPMD matmul kernel: (K,M) x (K,N) => (M,N).
    Distributes M dimension across 2 cores.
    """

    # 1) shapes
    K, M = lhsT.shape
    K2, N = rhs.shape
    assert K == K2, "lhsT and rhs must have the same contraction dimension"

    # 2) tile sizes
    TILE_M = nl.tile_size.gemm_stationary_fmax  # typically 128
    TILE_K = nl.tile_size.pmax                  # typically 128
    TILE_N = nl.tile_size.gemm_moving_fmax      # typically 512

    BLOCK_M = TILE_M * TILES_IN_BLOCK_M
    BLOCK_N = TILE_N * TILES_IN_BLOCK_N
    BLOCK_K = TILE_K * TILES_IN_BLOCK_K

    assert M % BLOCK_M == 0
    assert N % BLOCK_N == 0
    assert K % BLOCK_K == 0

    NUM_BLOCK_M = M // BLOCK_M
    NUM_BLOCK_N = N // BLOCK_N
    NUM_BLOCK_K = K // BLOCK_K

    # 3) figure out which slice of M blocks this spmd worker handles
    my_m_id = nl.program_id(0)             # 0 or 1 if spmd_m=2
    blocks_per_worker = NUM_BLOCK_M // spmd_m
    start_m = my_m_id * blocks_per_worker
    end_m   = (my_m_id + 1) * blocks_per_worker

    # 4) Allocate final result shape (M,N) in shared HBM
    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    # 5) Loop over the N dimension
    for bn_idx in nl.affine_range(NUM_BLOCK_N):
        # partial sums in SBUF for *our* M‑block slice
        partial_sbuf = nl.zeros(
            (blocks_per_worker, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N,
             nl.par_dim(TILE_M), TILE_N),
            dtype=lhsT.dtype,
            buffer=nl.sbuf
        )

        # 6) Loop over K blocks (sequential)
        for bk_idx in nl.sequential_range(NUM_BLOCK_K):
            # -- load tiles from RHS
            i_rhs = nl.mgrid[0:TILE_K, 0:BLOCK_N]
            rhs_tiles = nl.ndarray(
                (TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_N),
                dtype=rhs.dtype,
                buffer=nl.sbuf
            )
            for rsub in nl.affine_range(TILES_IN_BLOCK_K):
                rhs_tiles[rsub, i_rhs.p, i_rhs.x] = nl.load(
                    rhs[(TILES_IN_BLOCK_K * bk_idx + rsub)*TILE_K + i_rhs.p,
                        BLOCK_N * bn_idx + i_rhs.x]
                )

            # 7) Now loop over *our slice* of M blocks
            for local_m in nl.affine_range(blocks_per_worker):
                real_m = start_m + local_m

                # -- load tiles from LHS^T
                i_lhsT = nl.mgrid[0:TILE_K, 0:BLOCK_M]
                lhsT_tiles = nl.ndarray(
                    (TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_M),
                    dtype=lhsT.dtype,
                    buffer=nl.sbuf
                )
                for lsub in nl.affine_range(TILES_IN_BLOCK_K):
                    lhsT_tiles[lsub, i_lhsT.p, i_lhsT.x] = nl.load(
                        lhsT[(TILES_IN_BLOCK_K * bk_idx + lsub)*TILE_K + i_lhsT.p,
                             BLOCK_M * real_m + i_lhsT.x]
                    )

                # 8) Do the partial matmul accumulations
                i_lhsT_mm = nl.mgrid[0:TILE_K, 0:TILE_M]
                i_rhs_mm  = nl.mgrid[0:TILE_K, 0:TILE_N]
                i_res_mm  = nl.mgrid[0:TILE_M, 0:TILE_N]

                for bn_sub in nl.affine_range(TILES_IN_BLOCK_N):
                    for bm_sub in nl.affine_range(TILES_IN_BLOCK_M):
                        res_tile = nl.zeros((TILE_M, TILE_N),
                                            dtype=nl.float32,
                                            buffer=nl.psum)
                        for bk_sub in nl.affine_range(TILES_IN_BLOCK_K):
                            res_tile[...] += nisa.nc_matmul(
                                lhsT_tiles[bk_sub, i_lhsT_mm.p,
                                           bm_sub*TILE_M + i_lhsT_mm.x],
                                rhs_tiles[bk_sub, i_rhs_mm.p,
                                          bn_sub*TILE_N + i_rhs_mm.x]
                            )
                        partial_sbuf[local_m, bm_sub, bn_sub,
                                     i_res_mm.p, i_res_mm.x] += \
                            res_tile[i_res_mm.p, i_res_mm.x]

        # 9) Store final data from SBUF => result
        for local_m in nl.affine_range(blocks_per_worker):
            real_m = start_m + local_m
            for bm_sub in nl.affine_range(TILES_IN_BLOCK_M):
                i_res        = nl.mgrid[0:TILE_K, 0:TILE_N]
                i_res_packed = nl.mgrid[0:TILE_K, 0:BLOCK_N]
                result_packed = nl.ndarray((TILE_K, BLOCK_N),
                                           dtype=lhsT.dtype,
                                           buffer=nl.sbuf)
                # coalesce
                for bn_sub in nl.affine_range(TILES_IN_BLOCK_N):
                    result_packed[i_res.p,
                                  bn_sub*TILE_N + i_res.x] = nl.copy(
                        partial_sbuf[local_m, bm_sub, bn_sub,
                                     i_res.p, i_res.x]
                    )
                # store in final (M,N)
                nl.store(
                    result[(TILES_IN_BLOCK_M*real_m + bm_sub)*TILE_K + i_res_packed.p,
                           BLOCK_N*bn_idx + i_res_packed.x],
                    value=result_packed[i_res_packed.p, i_res_packed.x]
                )

    return result

@nki.jit
def matmul_test(lhs_small, rhs_small, size1, size2):
    nki_lhs_small = nl.load(lhs_small[:, :])
    nki_rhs_small = nl.load(rhs_small[:, :])
    # _, size1 = lhs_small.shape
    # _, size2 = rhs_small.shape
    res_psum = nl.zeros((size1, size2), nl.float32, buffer=nl.psum)  # 存储结果
    res_psum = nl.matmul(nki_lhs_small, nki_rhs_small, transpose_x=True)
    result = nl.ndarray((size1, size2), dtype=lhs_small.dtype, buffer=nl.shared_hbm)
    nl.store(result[:, :], value=res_psum)
    return result

@nki.jit
def fused_self_attn_for_SD_small_head_size(q_ref, k_ref, v_ref, use_causal_mask=False,
                                           mixed_precision=True):
    # Use q_ref dtype as the intermediate tensor dtype
    # Assume all IO tensors have the same dtype
    kernel_dtype = q_ref.dtype
    pe_in_dt = nl.bfloat16 if mixed_precision else np.float32
    assert q_ref.dtype == k_ref.dtype == v_ref.dtype

    # Shape checking
    seqlen, d_head = q_ref.shape
    assert d_head <= 128, "Cannot use this kernel for d_head > 128"
    assert tuple(q_ref.shape) == (seqlen, d_head), 'Input shape mismatch!'
    assert tuple(k_ref.shape) == (seqlen, d_head), 'Input shape mismatch!'
    assert tuple(v_ref.shape) == (seqlen,d_head), \
    f'Input shape mismatch! Expected: {(seqlen, d_head)} Actual: {tuple(v_ref.shape)}'
    out_ref = nl.ndarray((seqlen, d_head), dtype=q_ref.dtype, buffer=nl.shared_hbm)

    # Softmax scaling factor, multiplied onto Q
    softmax_scale = 0.125

    q_seq_n_tiles, q_seq_tile_size = seqlen // 128, 128
    k_seq_n_tiles, k_seq_tile_size = seqlen // 128, 128
    # No tiling on d_head dimension since the dimension of d_head fits in SB
    d_head_tile_size = d_head
    v_seq_n_tiles, v_seq_tile_size = seqlen // 128, 128

    ###################################
    # Step 1. transpose(tensor_v)
    ###################################
    # Buffer for v matrix transposed
    # Pre-fetch and keep it in SBUF throughout different softmax tiles
    trans_v = nl.ndarray((par_dim(v_seq_tile_size), v_seq_n_tiles, d_head), dtype=pe_in_dt)

    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
        ip_v = nl.arange(v_seq_tile_size)[:, None]
        if_v = nl.arange(d_head_tile_size)[None, :]
        trans_v[ip_v, i_k_seq_tile, if_v] = nl.load(
            v_ref[i_k_seq_tile * k_seq_tile_size + ip_v, if_v],
            dtype=pe_in_dt)

    q_local = nl.ndarray((q_seq_n_tiles, par_dim(d_head_tile_size), q_seq_tile_size), dtype=pe_in_dt)
    ip_q = nl.arange(d_head_tile_size)[:, None]
    if_q = nl.arange(q_seq_tile_size)[None, :]
    for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
        q_local[i_q_seq_tile, ip_q, if_q] = nl.load_transpose2d(
            q_ref[i_q_seq_tile * q_seq_tile_size + nl.arange(q_seq_tile_size)[:, None],
                    nl.arange(d_head_tile_size)[None, :]
            ],
            dtype=pe_in_dt) * softmax_scale

    k_local = nl.ndarray((k_seq_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size), dtype=pe_in_dt)
    ip_k = nl.arange(d_head_tile_size)[:, None]
    if_k = nl.arange(k_seq_tile_size)[None, :]
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
        k_local[i_k_seq_tile, ip_k, if_k] = nl.load_transpose2d(
            k_ref[i_k_seq_tile * k_seq_tile_size + nl.arange(k_seq_tile_size)[:, None],
            nl.arange(d_head_tile_size)[None, :]],
            dtype=pe_in_dt)

    for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):  # indent = 2
        # A SBUF buffer for an independent softmax tile
        qk_res_buf = nl.ndarray((par_dim(q_seq_tile_size), seqlen), dtype=kernel_dtype)

        neg_max_res = nl.ndarray((par_dim(q_seq_tile_size), k_seq_n_tiles), dtype=kernel_dtype)
        ip_max = nl.arange(q_seq_tile_size)[:, None]
        if_max = nl.arange(k_seq_n_tiles)[None, :]

        # Loop over RHS free of matmul(stationary=tensor_q, moving=tensor_k, contract=d_head)
        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):  # indent = 4

            # Since the K^T tile is the RHS, the q_seq_len dimension will be P in the result
            # PSUM buffer shape: [q_seq_tile_size P, k_seq_tile_size F]
            qk_psum = nl.zeros((par_dim(q_seq_tile_size), k_seq_tile_size),
                                dtype=np.float32, buffer=nl.psum)

            # Tensor indices for accessing qk result in k_seq_tile_size
            ip_qk = nl.arange(q_seq_tile_size)[:, None]
            if_qk = nl.arange(k_seq_tile_size)[None, :]

            ##############################################################
            # Step 2. matmul(stationary=tensor_q, moving=tensor_k, contract=d_head)
            ##############################################################
            qk_psum[ip_qk, if_qk] += nisa.nc_matmul(moving=k_local[i_k_seq_tile, ip_k, if_k],
                                                    stationary=q_local[i_q_seq_tile, ip_q, if_q])

            ###################################
            # Step 3. Apply optional causal mask
            ###################################
            if use_causal_mask:
                # Magic number -9984.0 to replace -inf similar to what neuronx-cc uses
                qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = nisa.affine_select(
                    pred=(i_q_seq_tile * q_seq_tile_size + ip_qk >= i_k_seq_tile * k_seq_tile_size + if_qk),
                    on_true_tile=qk_psum[ip_qk, if_qk], on_false_value=-9984.0, dtype=kernel_dtype)
            else:
                # Simply send psum result back to sbuf
                qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = nl.copy(qk_psum[ip_qk, if_qk],
                                                                              dtype=kernel_dtype)

            ###################################
            # Step 4. Softmax
            ###################################
            neg_max_res[ip_max, i_k_seq_tile] = nisa.tensor_reduce(
                np.max, data=qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk],
                axis=(1,), dtype=kernel_dtype, negate=True)

        neg_max_res_final = nisa.tensor_reduce(
            np.min, data=neg_max_res[ip_max, if_max],
            axis=(1,), dtype=kernel_dtype, negate=False)

        ip_softmax = nl.arange(q_seq_tile_size)[:, None]
        if_softmax = nl.arange(seqlen)[None, :]
        ip_sum_res = nl.arange(q_seq_tile_size)[:, None]
        if_sum_res = nl.arange(d_head_tile_size)[None, :]

        softmax_res = nl.ndarray((par_dim(q_seq_tile_size), seqlen), dtype=pe_in_dt)
        sum_divisor = nl.ndarray((par_dim(q_seq_tile_size), d_head_tile_size), dtype=kernel_dtype)

        # Simply use a large tile of seq_len in size since this is a "blocking" instruction
        # Assuming the compiler will merge exp and reduce_add into a single instruction on ACT
        exp_res = nisa.activation(np.exp,
                                data=qk_res_buf[ip_softmax, if_softmax],
                                bias=neg_max_res_final, scale=1.0)

        sum_res = nisa.tensor_reduce(np.add, data=exp_res, axis=(1,),
                            dtype=kernel_dtype)
        softmax_res[ip_softmax, if_softmax] = nl.copy(exp_res, dtype=pe_in_dt)

        sum_reciprocal_broadcast = (1.0 / sum_res).broadcast_to((q_seq_tile_size, d_head_tile_size))
        sum_divisor[ip_sum_res, if_sum_res] = nl.copy(sum_reciprocal_broadcast, dtype=kernel_dtype)

        # Buffer for transposed softmax results (FP32 in PSUM)
        trans_softmax_res = nl.ndarray(
            (par_dim(k_seq_tile_size), k_seq_n_tiles, q_seq_tile_size),
            dtype=pe_in_dt)

        # Result psum buffer has the hidden dim as P
        attn_res_psum = nl.zeros((par_dim(d_head_tile_size), q_seq_tile_size),
                                dtype=np.float32, buffer=nl.psum)

        ip_scores_t = nl.arange(k_seq_tile_size)[:, None]
        if_scores_t = nl.arange(q_seq_tile_size)[None, :]
        # Loop over matmul_1 contraction
        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            ###################################
            # Step 5. transpose(softmax_res)
            ###################################
            ip_scores = nl.arange(q_seq_tile_size)[:, None]
            if_scores = nl.arange(k_seq_tile_size)[None, :]

            trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t] = nisa.nc_transpose(
                softmax_res[ip_scores, i_k_seq_tile * k_seq_tile_size + if_scores])

        ip_out = nl.arange(d_head_tile_size)[:, None]
        if_out = nl.arange(q_seq_tile_size)[None, :]
        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            ######################################################################
            # Step 6. matmul_1(stationary=trans_v, moving=trans_softmax_res, contract=seqlen_v=seqlen_k)
            ######################################################################
            ip_v_t = nl.arange(k_seq_tile_size)[:, None]
            if_v_t = nl.arange(d_head_tile_size)[None, :]
            attn_res_psum[ip_out, if_out] += \
                nisa.nc_matmul(moving=trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t],
                            stationary=trans_v[ip_v_t, i_k_seq_tile, if_v_t])

        attn_res_sbuf = nl.copy(attn_res_psum[ip_out, if_out], dtype=kernel_dtype)

        attn_res_div = attn_res_sbuf * nisa.nc_transpose(sum_divisor[ip_sum_res, if_sum_res])

        nl.store(
        out_ref[i_q_seq_tile * q_seq_tile_size + if_out, ip_out],
        value=attn_res_div)

    return out_ref

@nki.jit
def tensor_transpose2D_kernel_(in_tensor, shape2D):
    out_tensor = nl.ndarray(in_tensor.shape, dtype=in_tensor.dtype,
                            buffer=nl.shared_hbm)
    # Gather input shapes
    sz_p, _ = in_tensor.shape

    # Load input data from external memory to on-chip memory
    in_tile = nl.load(in_tensor)

    # Performing f1/f2 transpose
    # ==========================
    # The desired transpose pattern is provided as an input:
    sz_f1, sz_f2 = shape2D

    # We're going to need 3 indices to perform f1:f2 transpose.
    # - i_p0 is the parallel index
    # - i_f1 and i_f2 are both free-dim indices, and will be used to transpose between the f1/f2 axes
    i_p0 = nl.arange(sz_p)[:, None, None]
    i_f1 = nl.arange(sz_f1)[None, :, None]
    i_f2 = nl.arange(sz_f2)[None, None, :]

    # Perform the transposition via a SBUF-to-SBUF copy, with access-pattern manipulation
    # Note that we have 2D tensors and 3 indices, since we need to represent a 2D access pattern *per partition*
    # RHS traverses an F1 x F2 matrix in a row major manner
    # LHS traverses an F2 x F1 (new) matrix in a row major manner
    out_tile = nl.ndarray(shape=(sz_p, sz_f2*sz_f1), dtype=out_tensor.dtype)
    out_tile[i_p0, i_f2*sz_f1+i_f1] = nl.copy(in_tile[i_p0, i_f1*sz_f2+i_f2])

    # Finally, we store out_tile to external memory
    nl.store(out_tensor, value=out_tile)

    return out_tensor

@nki.jit
def nki_matmul_basic_(lhsT, rhs):
    """NKI kernel to compute a 64x128x512 matrix multiplication operation

    Args:
        lhsT: an input tensor of shape [128,64], a left hand side argument of the
        matrix multiplication, delivered transposed for optimal performance
        rhs: an input tensor of shape [128,512], a right hand side argument of the
        matrix multiplication
    Returns:
        result: the resulting output tensor of shape [64,512]
    """
    # result = nl.ndarray((64, 512), dtype=lhsT.dtype, buffer=nl.shared_hbm)
    result = nl.ndarray((32, 8192), dtype=lhsT.dtype, buffer=nl.shared_hbm)


    # Defining indexes for input LHS.T
    # - Note: here we take LayoutConstraint #1 into account:
    # "For MatMult, contraction axis must be mapped to P-dim"
    i_lhsT_p, i_lhsT_f = nl.mgrid[0:2048, 0:32]

    # Defining indexes for input RHS
    # - Note: here we take LayoutConstraint #1 into account:
    # "For MatMult, contraction axis must be mapped to P-dim"
    i_rhs_p, i_rhs_f = nl.mgrid[0:2048, 0:8192]

    # Defining indexes for the output ([64,128]@[128,512] -> [64,512])
    i_out_p, i_out_f = nl.mgrid[0:32, 0:8192]

    # Loading the inputs (HBM->SBUF)
    # Note: here we take Tile dtype definition into account,
    # which forces P-dim as the left most index
    lhs_tile = nl.load(lhsT[i_lhsT_p, i_lhsT_f])
    rhs_tile = nl.load(rhs[i_rhs_p, i_rhs_f])

    # Perform the matrix-multiplication
    # Note1: We set transpose_x to True, to indicate that the LHS input is transposed
    # Note2: A NKI matmul instruction always writes to PSUM in float32 data-type
    result_psum = nl.matmul(lhs_tile, rhs_tile, transpose_x=True)

    # Copy the result from PSUM back to SBUF, and cast to expected output data-type
    result_sbuf = nl.copy(result_psum, dtype=result.dtype)

    # The result of a [64,128] x [128,512] matrix multiplication has a shape of [64, 512].
    # This dictates which indices to use to address the result tile.
    nl.store(result[i_out_p, i_out_f], value=result_sbuf)

    return result

@nki.jit
def nki_matmul_fully_optimized_(
    lhsT,
    rhs,
    TILES_IN_BLOCK_M=1,
    TILES_IN_BLOCK_N=16,
    TILES_IN_BLOCK_K=16,
):
    K, M = lhsT.shape
    K_, N = rhs.shape
    assert K == K_, "lhsT and rhs must have the same contraction dimension"
    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    #TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    
    # FY:
    TILE_M = min(nl.tile_size.gemm_stationary_fmax, M)
    TILE_K = nl.tile_size.pmax  # 128
    TILE_N = nl.tile_size.gemm_moving_fmax  # 512

    BLOCK_M = TILE_M * TILES_IN_BLOCK_M
    BLOCK_N = TILE_N * TILES_IN_BLOCK_N
    BLOCK_K = TILE_K * TILES_IN_BLOCK_K

    # the size has to be multiple of block size
    assert M % BLOCK_M == 0
    assert N % BLOCK_N == 0
    assert K % BLOCK_K == 0

    NUM_BLOCK_M = M // BLOCK_M
    NUM_BLOCK_N = N // BLOCK_N
    NUM_BLOCK_K = K // BLOCK_K

    # Blocking N dimension (the RHS free dimension)
    for n in nl.affine_range(NUM_BLOCK_N):
        result_tiles = nl.zeros((NUM_BLOCK_M, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N,
            nl.par_dim(TILE_M), TILE_N),
            dtype=lhsT.dtype,
            buffer=nl.sbuf)

        # Blocking K dimension (the contraction dimension)
        # Use `sequential_range` because we do not want the compiler to change this loop by, 
        # for example, vectorizing it
        for k in nl.sequential_range(NUM_BLOCK_K):
            # Loading tiles from rhs
            # setting the load tile to `TILE_K x BLOCK_SIZE_N` to optimize DMA performance
            i_rhs = nl.mgrid[0:TILE_K, 0:BLOCK_N]
            rhs_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_N),
                                    dtype=rhs.dtype,
                                    buffer=nl.sbuf)

            for bk_r in nl.affine_range(TILES_IN_BLOCK_K):
                rhs_tiles[bk_r, i_rhs.p, i_rhs.x] = nl.load(
                    rhs[(TILES_IN_BLOCK_K * k + bk_r) * TILE_K + i_rhs.p,
                    BLOCK_N * n + i_rhs.x])

            # Blocking M dimension (the LHS free dimension)
            for m in nl.affine_range(NUM_BLOCK_M):
            # Loading tiles from lhsT
                i_lhsT = nl.mgrid[0:TILE_K, 0:BLOCK_M]
                lhsT_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_M),
                                        dtype=lhsT.dtype,
                                        buffer=nl.sbuf)
                for bk_l in nl.affine_range(TILES_IN_BLOCK_K):
                    lhsT_tiles[bk_l, i_lhsT.p, i_lhsT.x] = nl.load(
                        lhsT[(TILES_IN_BLOCK_K * k + bk_l) * TILE_K + i_lhsT.p,
                        BLOCK_M * m + i_lhsT.x])

                # Do matmul with all tiles in the blocks
                i_lhsT_mm = nl.mgrid[0:TILE_K, 0:TILE_M]
                i_rhs_mm = nl.mgrid[0:TILE_K, 0:TILE_N]
                i_res_mm = nl.mgrid[0:TILE_M, 0:TILE_N]
                for bn in nl.affine_range(TILES_IN_BLOCK_N):
                    for bm in nl.affine_range(TILES_IN_BLOCK_M):
                        res_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

                        for bk in nl.affine_range(TILES_IN_BLOCK_K):
                            res_tile[...] += nisa.nc_matmul(
                                lhsT_tiles[bk, i_lhsT_mm.p, bm * TILE_M + i_lhsT_mm.x],
                                rhs_tiles[bk, i_rhs_mm.p, bn * TILE_N + i_rhs_mm.x])

                        # Accumulate on corresponding SBUF tile
                        result_tiles[m, bm, bn, i_res_mm.p,
                            i_res_mm.x] += res_tile[i_res_mm.p, i_res_mm.x]

        # Copying the result from SBUF to HBM
        for m in nl.affine_range(NUM_BLOCK_M):
            for bm in nl.affine_range(TILES_IN_BLOCK_M):
                i_res = nl.mgrid[0:TILE_K, 0:TILE_N]
                i_res_packed = nl.mgrid[0:TILE_K, 0:BLOCK_N]
                result_packed = nl.ndarray((TILE_K, BLOCK_N),
                    dtype=result_tiles.dtype,
                    buffer=nl.sbuf)

                # coalesce result tiles for better DMA performance
                for bn in nl.affine_range(TILES_IN_BLOCK_N):
                    result_packed[i_res.p,
                        bn * TILE_N + i_res.x] = nl.copy(result_tiles[m, bm, bn,
                                                                    i_res.p,
                                                                    i_res.x])
                nl.store(result[(TILES_IN_BLOCK_M * m + bm) * TILE_K + i_res_packed.p,
                                BLOCK_N * n + i_res_packed.x],
                            value=result_packed[i_res_packed.p, i_res_packed.x])

    return result

@nki.jit
def nki_silu_mul(lhs_tensor, rhs_tensor):
    """
    Tile-based SiLU + Multiply kernel, using nl.mgrid for 2D indexing.
    """
    # 1) Validate shape
    H, W = lhs_tensor.shape
    assert lhs_tensor.shape == rhs_tensor.shape, (
        f"Shape mismatch: {lhs_tensor.shape} vs {rhs_tensor.shape}"
    )
    TILE_H = 128
    TILE_W = 512
    assert H % TILE_H == 0, f"H={H} not multiple of {TILE_H}"
    assert W % TILE_W == 0, f"W={W} not multiple of {TILE_W}"

    # 2) Allocate final output in HBM
    out_tensor = nl.ndarray((H, W), dtype=lhs_tensor.dtype, buffer=nl.shared_hbm)

    num_tiles_h = H // TILE_H
    num_tiles_w = W // TILE_W

    # 3) Loop over tiles
    for tile_h_idx in nl.affine_range(num_tiles_h):
        start_h = tile_h_idx * TILE_H

        for tile_w_idx in nl.affine_range(num_tiles_w):
            start_w = tile_w_idx * TILE_W

            # Allocate a tile in ephemeral memory: shape (128, 512)
            out_nl_tile = nl.zeros((TILE_H, TILE_W), dtype=lhs_tensor.dtype, buffer=nl.sbuf)

            # 4) Use nl.mgrid to index over [0..TILE_H) x [0..TILE_W)
            #    .p is the first dimension, .x is the second dimension
            i_rc = nl.mgrid[0:TILE_H, 0:TILE_W]

            lhs_val  = nl.load(lhs_tensor[start_h + i_rc.p, start_w + i_rc.x])
            lhs_silu = nl.silu(lhs_val)
            rhs_val  = nl.load(rhs_tensor[start_h + i_rc.p, start_w + i_rc.x])

            # Store elementwise multiplication in the ephemeral tile
            out_nl_tile[i_rc.p, i_rc.x] = nl.multiply(lhs_silu, rhs_val)

            # 5) Store ephemeral tile to final HBM output
            nl.store(
                out_tensor[start_h : start_h + TILE_H, start_w : start_w + TILE_W],
                value=out_nl_tile
            )

    return out_tensor

@nki.jit
def nki_rmsnorm_kernel(a_tensor, g_tensor, eps):
    # Calculate out_tensor = a_tensor/RMS(a_tensor) * g_tensor
    # Where RMS(a_tensor) = sqrt((1/N) * sum(a_tensor * a_tensor))
    # and N = a_tensor.shape[1]
    # Reduction (mean) is performed in the free (2nd) dimension

    out_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype,
                          buffer=nl.shared_hbm)

    # Make sure shapes match
    assert a_tensor.shape[2] == g_tensor.shape[0]

    # Generate tensor indices to index input tensor
    ix = nl.arange(128)[:, None]
    iw = nl.arange(1)[:, None]
    iy = nl.arange(a_tensor.shape[2])[None, :]

    num_rows = a_tensor.shape[1]

    # Load RMSNorm weight once, reused by rows/tiles of a_tensor
    g_tile = nl.load(g_tensor.reshape((1, g_tensor.shape[0]))[iw, iy])

    # Process 128 rows at a time due to 128-partition tile size limitation
    # Since we're not reducing across the first dimension
    # Tiles can be processed independently

    for b in range(a_tensor.shape[0]):
        for i in range(math.ceil(a_tensor.shape[1]/128)):
            # Load input data from external memory to on-chip memory
            a_tile = nl.zeros([128, a_tensor.shape[2]], a_tensor.dtype)
            # print("a tensor shape2", a_tensor.shape[2])
            a_tile[...] = nl.load(a_tensor[b, i * 128 + ix, iy], mask=(i * 128 + ix < num_rows))

            # Compute element-wise square of a_tensor
            in_square = nl.square(a_tile)

            # Calculate sum of squared elements, along last dimension
            square_sum = nl.sum(in_square, axis=[1])

            # Scale and get a reciprocal
            mean = square_sum / a_tensor.shape[2]

            # Take square root of mean and then reciprocal with
            # rsqrt API (one ISA instruction)
            rms_reciprocal = nl.rsqrt(mean + eps)

            # Scale the input tensor
            out_tile = nl.multiply(a_tile, rms_reciprocal)

            # Broadcast weight along first axis to match tensor shape
            # num_rows_active = min(num_rows - i * 128, 128)
            g_bcast = g_tile.broadcast_to((128, g_tensor.shape[0]))

            # Multiply with the RMSNorm weight
            out_tile[...] = nl.multiply(out_tile, g_bcast, mask=(i * 128 + ix < num_rows))

            # store the addition results back to external memory (out_tensor)
            nl.store(out_tensor[b, i * 128 + ix, iy], value=out_tile, mask=(i * 128 + ix < num_rows))

    return out_tensor

def block_matmul(A, B, block_size=512, block_size_1=128):
    """
    分块矩阵乘法函数，适用于大矩阵乘法，使用 PyTorch 实现。

    参数:
    A (torch.Tensor): 第一个矩阵，形状为 [m, k]
    B (torch.Tensor): 第二个矩阵，形状为 [k, n]
    block_size (int): 分块大小，默认为 128

    返回:
    torch.Tensor: 结果矩阵，形状为 [m, n]
    """
    m, k = A.shape
    k, n = B.shape
    # original_dtype = A.dtype
    # A = A.to(torch.float32)
    # B = B.to(torch.float32)
    # device = xm.xla_device()
    # A = A.to(device)
    # B = B.to(device)

    # 初始化结果矩阵
    result = torch.zeros((m, n), dtype = A.dtype, device=A.device)

    # 分块计算矩阵乘法
    for i in range(0, m, block_size_1):
        for j in range(0, n, block_size):
            for l in range(0, k, block_size_1):
                # 获取当前块
                A_block = A[i:i+block_size_1, l:l+block_size_1]
                B_block = B[l:l+block_size_1, j:j+block_size]
                m1, k1 = A_block.shape
                k1, n1 = B_block.shape

                # 计算当前块的乘积
                result[i:i+block_size_1, j:j+block_size] += matmul_test(A_block.T, B_block, m1, n1)
    # result = result.to(original_dtype)
    return result


class CustomRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, nki_enabled=False):
        """
        Use this RMSNorm to perform customized rmsnorm on Neuron
        Note: CustomRMSNorm forward method calls target="AwsNeuronRmsNorm"
        """
        super().__init__()
        self.weight = nn.Parameter(ones(hidden_size))
        self.variance_epsilon = eps
        self.nki_enabled = nki_enabled

    def forward(self, hidden_states):
        # if self.nki_enabled:
        #     out_tensor = nki_rmsnorm_kernel(hidden_states, self.weight, self.variance_epsilon)
        #     return out_tensor

        original_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        # print("rms_test:", hidden_states.size())
        # print("weights_size:", self.weight.size())
        result = RmsNorm.apply(
            hidden_states, self.weight, self.variance_epsilon, len(hidden_states.shape) - 1
        )

        return result.to(original_dtype)


def get_rmsnorm_cls():
    # Initialize to the appropriate implementation of RMSNorm
    # If infer on NXD -> CustomRMSNorm
    # If infer on CPU -> HF_RMSNorm (CustomRMSNorm does not work on CPU)
    return CustomRMSNorm if parallel_state.model_parallel_is_initialized() else LlamaRMSNorm


def preshard_hook_fn(module: torch.nn.Module, model_state_dict: dict, prefix: str) -> bool:
    if isinstance(module, (BaseGroupQueryAttention,)):
        return module.preshard_hook(model_state_dict, prefix)

    return False


def _register_module(key: str, cls: Type[nn.Module]):
    _LLAMA_MODULE_MAP[key] = cls


def register_module(key: str):
    """
    Register a module for use in NeuronLlama.

    Arguments:
        key: String used to identify the module

    Example:
        @register_module("NeuronLlamaAttention")
        class NeuronLlamaAttention(nn.Module):
            ...
    """

    def inner(cls: Type[nn.Module]):
        _register_module(key, cls)
        return cls

    return inner


def convert_state_dict_to_fused_qkv(llama_state_dict, cfg: InferenceConfig):
    """
    This function concats the qkv weights to a Wqkv weight for fusedqkv, and deletes the qkv weights.
    """
    for l in range(cfg.num_hidden_layers):  # noqa: E741
        llama_state_dict[f"layers.{l}.self_attn.Wqkv.weight"] = torch.cat(
            [
                llama_state_dict[f"layers.{l}.self_attn.q_proj.weight"],
                llama_state_dict[f"layers.{l}.self_attn.k_proj.weight"],
                llama_state_dict[f"layers.{l}.self_attn.v_proj.weight"],
            ],
        )
        del llama_state_dict[f"layers.{l}.self_attn.q_proj.weight"]
        del llama_state_dict[f"layers.{l}.self_attn.k_proj.weight"]
        del llama_state_dict[f"layers.{l}.self_attn.v_proj.weight"]

    gc.collect()

    return llama_state_dict


class NeuronConfigNKI(NeuronConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nki_enabled = kwargs.pop("enable_nki", False)


class LlamaInferenceConfig(InferenceConfig):
    def add_derived_config(self):
        self.num_cores_per_group = 1
        if self.neuron_config.flash_decoding_enabled:
            num_attn_heads, num_kv_heads = self.num_attention_heads, self.num_key_value_heads
            self.num_cores_per_group = calculate_num_cores_per_group(
                num_attn_heads, num_kv_heads, self.neuron_config.tp_degree
            )

    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size",
            "num_attention_heads",
            "num_hidden_layers",
            "num_key_value_heads",
            "pad_token_id",
            "vocab_size",
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",
            "hidden_act",
        ]

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:
        return NeuronConfigNKI


class NeuronLlamaMLP(nn.Module):
    """
    This class just replace the linear layers (gate_proj, up_proj and down_proj) with column and row parallel layers
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()

        self.config = config
        self.neuron_config = config.neuron_config
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]

        self.sequence_parallel_enabled = getattr(
            self.neuron_config, "sequence_parallel_enabled", False
        )
        self.sequence_dimension = 1 if self.sequence_parallel_enabled else None
        self.rms_norm_eps = config.rms_norm_eps
        self.mlp_kernel_enabled = self.neuron_config.mlp_kernel_enabled
        self.quantized_mlp_kernel_enabled = self.neuron_config.quantized_mlp_kernel_enabled
        self.rmsnorm_quantize_kernel_enabled = self.neuron_config.rmsnorm_quantize_kernel_enabled
        self.quantized_kernel_lower_bound = self.neuron_config.quantized_kernel_lower_bound
        self.logical_neuron_cores = self.neuron_config.logical_neuron_cores
        mlp_bias = getattr(config, "mlp_bias", False)
        self.mlp_bias = mlp_bias

        if parallel_state.model_parallel_is_initialized():
            if self.quantized_mlp_kernel_enabled:
                # Quantized MLP kernels expect intermediate size to be multiple of 128, so we need to pad
                tp_degree = self.neuron_config.tp_degree
                self.intermediate_size += (
                    get_padding_length(self.intermediate_size // tp_degree, 128) * tp_degree
                )
                # logger.debug(f"Quantized intermediate_size: {self.intermediate_size}")

                quantization_type = QuantizationType(self.neuron_config.quantization_type)
                quantized_dtype = QuantizedDtype.F8E4M3
                self.gate_proj = QuantizedColumnParallel(
                    input_size=self.hidden_size,
                    output_size=self.intermediate_size,
                    bias=mlp_bias,
                    gather_output=False,
                    sequence_parallel_enabled=False,
                    dtype=config.neuron_config.torch_dtype,
                    quantized_dtype=quantized_dtype,
                    quantization_type=quantization_type,
                    tensor_model_parallel_group=get_tp_group(config),
                )
                self.up_proj = QuantizedColumnParallel(
                    input_size=self.hidden_size,
                    output_size=self.intermediate_size,
                    bias=mlp_bias,
                    gather_output=False,
                    sequence_parallel_enabled=False,
                    dtype=config.neuron_config.torch_dtype,
                    quantized_dtype=quantized_dtype,
                    quantization_type=quantization_type,
                    tensor_model_parallel_group=get_tp_group(config),
                )
                self.down_proj = QuantizedRowParallel(
                    input_size=self.intermediate_size,
                    output_size=self.hidden_size,
                    bias=mlp_bias,
                    quantization_type=quantization_type,
                    input_is_parallel=True,
                    dtype=config.neuron_config.torch_dtype,
                    quantized_dtype=quantized_dtype,
                    sequence_parallel_enabled=False,
                    quantization_per_channel_axis=0,
                    tensor_model_parallel_group=get_tp_group(config),
                )

            else:
                # print("initialize columnxxx")
                self.gate_proj = ColumnParallelLinear(
                    self.hidden_size,
                    self.intermediate_size,
                    bias=mlp_bias,
                    gather_output=False,
                    dtype=config.neuron_config.torch_dtype,
                    pad=True,
                    sequence_parallel_enabled=False,
                    sequence_dimension=None,
                    tensor_model_parallel_group=get_tp_group(config),
                )
                self.up_proj = ColumnParallelLinear(
                    self.hidden_size,
                    self.intermediate_size,
                    bias=mlp_bias,
                    gather_output=False,
                    dtype=config.neuron_config.torch_dtype,
                    pad=True,
                    sequence_parallel_enabled=False,
                    sequence_dimension=None,
                    tensor_model_parallel_group=get_tp_group(config),
                )
                self.down_proj = RowParallelLinear(
                    self.intermediate_size,
                    self.hidden_size,
                    bias=mlp_bias,
                    input_is_parallel=True,
                    dtype=config.neuron_config.torch_dtype,
                    pad=True,
                    sequence_parallel_enabled=self.sequence_parallel_enabled,
                    sequence_dimension=self.sequence_dimension,
                    tensor_model_parallel_group=get_tp_group(config),
                    reduce_dtype=config.neuron_config.rpl_reduce_dtype,
                )
                # self.gate_proj.weight.data = transpose_parallel_linear_layer(self.gate_proj.weight.data)
                # self.up_proj.weight.data   = transpose_parallel_linear_layer(self.up_proj.weight.data)
                # self.down_proj.weight.data = transpose_parallel_linear_layer(self.down_proj.weight.data)

            if self.mlp_kernel_enabled:
                if self.quantized_mlp_kernel_enabled:
                    preprocess_quantized_linear_layer(self.gate_proj)
                    preprocess_quantized_linear_layer(self.up_proj)
                    preprocess_quantized_linear_layer(self.down_proj)

                else:
                    # Transpose the weights to the layout expected by kernels
                    # self.gate_proj.weight = transpose_parallel_linear_layer(self.gate_proj.weight)
                    # self.up_proj.weight = transpose_parallel_linear_layer(self.up_proj.weight)
                    # self.down_proj.weight = transpose_parallel_linear_layer(self.down_proj.weight)
                    pass

        else:
            # 如果没法并行就直接用pytorch提供的线性层
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=mlp_bias)

    def _kernel_enabled_quantized_mlp(self, x, fused_rmsnorm, rmsnorm, residual, adapter_ids):
        grid = (vnc(self.logical_neuron_cores),)
        fused_residual = residual is not None
        # logger.debug(
        #     f"MLP: quantized kernel, fused_residual={fused_residual}, fused_rmsnorm={fused_rmsnorm}, logical_neuron_cores={self.logical_neuron_cores}"
        # )

        # Can't do residual add in the kernel if SP is enabled
        if fused_residual:
            assert (
                not self.sequence_parallel_enabled
            ), "Quantized MLP cannot have both fused residual add and sequence parallel RMSnorm!"
            # Using fused residual add
            _mlp_fwd_call = nki_jit()(quant_mlp_fused_add_isa_kernel)
        else:
            _mlp_fwd_call = nki_jit()(quant_mlp_isa_kernel)

        # Handle SP RMSnorm
        x_orig_dtype = x.dtype
        if self.sequence_parallel_enabled:
            # This RMSNormQuant kernel will do quantization inside, so we pass the
            # lower_bound for clipping.
            # If we don't use this kernel, the MLP kernel below will do the
            # quantization, so we also pass lower_bound to that kernel.
            if self.rmsnorm_quantize_kernel_enabled:
                # logger.debug(
                #     "Running Quantized MLP kernel with sequence-parallel RMSnorm-Quantize kernel!"
                # )
                _rmsnorm_quant_fwd_call = nki_jit()(rmsnorm_quant_isa_kernel)
                quant_rmsnorm_out = torch.zeros(
                    size=(
                        x.shape[0],  # batch size
                        x.shape[1],  # sequence length
                        x.shape[2] + 4,  # hidden size + 4 bytes for packing fp32 scale
                    ),
                    dtype=torch.int8,
                    device=x.device,
                )
                ln_w = rmsnorm.weight.unsqueeze(0)
                lower_bound = self.quantized_kernel_lower_bound
                _rmsnorm_quant_fwd_call[grid](
                    x, ln_w, lower_bound, quant_rmsnorm_out, kernel_name="QuantOnly"
                )
                x = gather_from_sequence_parallel_region(
                    quant_rmsnorm_out,
                    self.sequence_dimension,
                    process_group=get_tp_group(self.config),
                )

            else:
                # logger.debug(
                #     "Running Quantized MLP kernel with external (native compiler) sequence-parallel RMSnorm!"
                # )
                x = gather_from_sequence_parallel_region(
                    x, self.sequence_dimension, process_group=get_tp_group(self.config)
                )

        # Build output tensor
        output_tensor_seqlen = x.shape[1]
        if fused_residual:
            # seqlen dim is doubled to store the residual add output
            output_tensor_seqlen *= 2

        output_tensor = torch.zeros(
            size=(
                x.shape[0],  # batch size
                output_tensor_seqlen,
                self.hidden_size,  # hidden size
            ),
            dtype=x_orig_dtype,
            device=x.device,
        )

        # Grab weights
        # all weights of the layers are stored in (out, in) shape
        # unsqueeze so that shape of RMS gamma weight is [1, hidden] instead of [hidden]
        ln_w = rmsnorm.weight.unsqueeze(0)
        gate_w = self.gate_proj.weight.data
        gate_w_scale = self.gate_proj.weight_scale
        up_w = self.up_proj.weight.data
        up_w_scale = self.up_proj.weight_scale
        down_w = self.down_proj.weight.data
        down_w_scale = self.down_proj.weight_scale
        lower_bound = self.quantized_kernel_lower_bound

        if fused_residual:
            _mlp_fwd_call[grid](
                x,  # attn_output
                residual,  # hidden
                ln_w,  # ln_w
                gate_w,  # gate_w
                gate_w_scale,
                up_w,  # up_w
                up_w_scale,
                down_w,  # down_w
                down_w_scale,
                lower_bound,
                output_tensor,  # out
                fused_rmsnorm=fused_rmsnorm,
                eps=self.rms_norm_eps,
                kernel_name="MLP",
                store_add=True,
            )
            original_seqlen = x.shape[1]
            residual = output_tensor[:, original_seqlen:, :]
            output_tensor = output_tensor[:, :original_seqlen, :]
        else:
            _mlp_fwd_call[grid](
                x,  # hidden
                # should be fine to pass gamma is as a dummy even if not using fused rmsnorm
                ln_w,
                gate_w,  # gate_w
                gate_w_scale,
                up_w,  # up_w
                up_w_scale,
                down_w,  # down_w
                down_w_scale,
                lower_bound,
                output_tensor,  # out
                # Run RMSNorm inside the kernel if NOT using SP rmsnorm
                fused_rmsnorm=fused_rmsnorm,
                eps=self.rms_norm_eps,
                kernel_name="MLP",
            )
            residual = None

        # All-reduce or reduce-scatter, depending on whether SP is enabled
        if self.sequence_parallel_enabled:
            output_tensor = reduce_scatter_to_sequence_parallel_region(
                output_tensor, self.sequence_dimension, process_group=get_tp_group(self.config)
            )
        else:
            output_tensor = reduce_from_tensor_model_parallel_region(output_tensor)

        # logger.debug(f"Quantized MLP output shape {output_tensor.shape}")
        return (output_tensor, residual)

    def _kernel_enabled_mlp(self, x, fused_rmsnorm, rmsnorm, residual, adapter_ids):
        fused_residual = residual is not None
        # logger.debug(
        #     f"MLP: kernel, fused_residual={fused_residual}, fused_rmsnorm={fused_rmsnorm}, logical_neuron_cores={self.logical_neuron_cores}"
        # )

        # Choose which kernel to call
        if fused_residual:
            assert (
                not self.sequence_parallel_enabled
            ), "MLP kernel cannot have both fused residual add and sequence parallel RMSnorm!"
            # Using fused residual add
            _mlp_fwd_call = nki_jit()(mlp_fused_add_isa_kernel)
        else:
            _mlp_fwd_call = nki_jit()(mlp_isa_kernel)

        if self.sequence_parallel_enabled:
            x = gather_from_sequence_parallel_region(
                x, self.sequence_dimension, process_group=get_tp_group(self.config)
            )

        # Build output tensor
        output_tensor_seqlen = x.shape[1]
        if fused_residual:
            # seqlen dim is doubled to store the residual add output
            output_tensor_seqlen *= 2

        output_tensor = torch.zeros(
            size=(
                x.shape[0],  # batch size
                output_tensor_seqlen,
                self.hidden_size,  # hidden size
            ),
            dtype=x.dtype,
            device=x.device,
        )

        # Grab weights
        # all weights of the layers are stored in (out, in) shape
        # unsqueeze so that shape of RMS gamma weight is [1, hidden] instead of [hidden]
        ln_w = rmsnorm.weight.unsqueeze(0)
        gate_w = self.gate_proj.weight.data
        up_w = self.up_proj.weight.data
        down_w = self.down_proj.weight.data

        grid = (vnc(self.logical_neuron_cores),)

        if fused_residual:
            _mlp_fwd_call[grid](
                x,  # attn_output
                residual,  # hidden
                ln_w,  # ln_w
                gate_w,  # gate_w
                up_w,  # up_w
                down_w,  # down_w
                output_tensor,  # out
                fused_rmsnorm=fused_rmsnorm,
                eps=self.rms_norm_eps,
                kernel_name="MLP",
                store_add=True,
            )
            original_seqlen = x.shape[1]
            residual = output_tensor[:, original_seqlen:, :]
            output_tensor = output_tensor[:, :original_seqlen, :]
        else:
            _mlp_fwd_call[grid](
                x,  # hidden
                # should be fine to pass gamma is as a dummy even if not using fused rmsnorm
                ln_w,
                gate_w,
                up_w,
                down_w,
                output_tensor,  # out
                # Run RMSNorm inside the kernel if NOT using SP rmsnorm
                fused_rmsnorm=fused_rmsnorm,
                eps=self.rms_norm_eps,
                kernel_name="MLP",
            )
            residual = None

        # All-reduce or reduce-scatter, depending on whether SP is enabled
        if self.sequence_parallel_enabled:
            output_tensor = reduce_scatter_to_sequence_parallel_region(
                output_tensor, self.sequence_dimension, process_group=get_tp_group(self.config)
            )
        else:
            output_tensor = reduce_from_tensor_model_parallel_region(
                output_tensor, process_group=get_tp_group(self.config)
            )

        # logger.debug(f"MLP output shape {output_tensor.shape}")
        return (output_tensor, residual)

    def _native_mlp(self, x, rmsnorm, adapter_ids=None):

        if self.sequence_parallel_enabled:
            x = gather_from_sequence_parallel_region(
                x, self.sequence_dimension, process_group=get_tp_group(self.config)
            )

        ## 可以操作
        # print(x.size())
        # print("weight:", self.up_proj.weight.T.size())
        # bs, head, dimension = x.size()
        # head_w, dimension_w = self.up_proj.weight.T.size()
        # result = torch.zeros((bs, head, dimension_w), dtype=x.dtype, device=x.device)
        # for i in range(bs):
        #     result[i] = block_matmul(x[i], self.gate_proj.weight.T)
        #     # print("temp:", temp.size())
        #     # print("result[i]:", result[i].size())
        # gate_proj_output = result


######################################################################################################################################################

        gate_proj_output = torch.matmul(x, self.gate_proj.weight.T)
        up_proj_output = torch.matmul(x, self.up_proj.weight.T)

######################################################################################################################################################

        # B, N, C = x.shape   # For example, B * N = 32, C = 2048
        # M_orig = B * N      # Original M dimension (32)

        # # We need M to be a multiple of 256 = 128 x 2core.
        # BLOCK_M = 256
        # M_pad = math.ceil(M_orig / BLOCK_M) * BLOCK_M  # Next multiple of 256

        # # Flatten x to 2D: [B*N, C]
        # x_flat_padded = x.view(M_orig, C) # was called x_flat
        # gate_weight_T = self.gate_proj.weight.T
        # up_weight_T = self.up_proj.weight.T
        # # Pad x_flat along dimension 0 if needed.
        # if M_pad > M_orig:
        #     pad_rows = M_pad - M_orig
        #     pad = torch.zeros(pad_rows, C, dtype=x.dtype, device=x.device)
        #     x_flat_padded = torch.cat([x_flat_padded, pad], dim=0)

        # x_flat_padded_T = x_flat_padded.T
        # MDIM = x_flat_padded.shape[0]
        # NDIM = self.gate_proj.weight.shape[0]
        # KDIM = self.gate_proj.weight.shape[1]

        # Mtile = MDIM // 128 // 2 ### 2 cores
        # Ntile = NDIM // 512
        # Ktile = KDIM // 128 

        # # gate_proj_output = (
        # #     self.gate_proj(x)
        # #     if not is_lora_module(self.gate_proj)
        # #     else self.gate_proj(x, adapter_ids)
        # # )
        # output_padded = nki_matmul_fully_optimized_spmd_[nl.nc(2)](x_flat_padded_T, gate_weight_T, Mtile, Ntile, Ktile, 2)
        # output_flat = output_padded[:M_orig, :]
        # gate_proj_output = output_flat.view(B, N, -1)

        # # up_proj_output = (
        # #     self.up_proj(x) if not is_lora_module(self.up_proj) else self.up_proj(x, adapter_ids)
        # # )
        # up_output_padded = nki_matmul_fully_optimized_spmd_[nl.nc(2)](x_flat_padded_T, up_weight_T, Mtile, Ntile, Ktile, 2)        
        # up_output_flat = up_output_padded[:M_orig, :]
        # up_proj_output = up_output_flat.view(B, N, -1)

######################################################################################################################################################

        # gate_weight = self.gate_proj.weight.T
        # up_weight = self.up_proj.weight.T

        # BLOCK_M = 256        # or whatever multiple of 128 * #cores you use

        # B, N_, C = x.shape
        # M_orig = B * N_          # e.g. 32 in your example

        # # We need M to be multiple of 256=128x2 for 2-core parallel
        # M_pad = math.ceil(M_orig / BLOCK_M) * BLOCK_M

        # # Flatten x to [M_orig, C]
        # x_flat = x.view(M_orig, C)

        # if M_pad > M_orig:
        #     pad_rows = M_pad - M_orig
        #     pad = torch.zeros(pad_rows, C, dtype=x.dtype, device=x.device)
        #     x_flat_padded = torch.cat([x_flat, pad], dim=0)
        # else:
        #     x_flat_padded = x_flat

        # # Transpose to match kernel’s shape (K= C, M= M_pad)
        # x_flat_padded_T = x_flat_padded.T  # shape (C, M_pad)

        # # -- Horizontally concat gate & up weight: shape (C, out1 + out2)
        # combined_weight = torch.cat([gate_weight, up_weight], dim=1)
        # out1 = gate_weight.shape[1]
        # out2 = up_weight.shape[1]
        # assert combined_weight.shape == (C, out1 + out2)

        # # Decide tiling parameters exactly as you do now
        # MDIM = x_flat_padded.shape[0]    # = M_pad
        # KDIM = C                         # same as kernel "K"
        # NDIM = out1 + out2              # total columns

        # # e.g. for 2 cores, tile M dim in multiples of 128, etc.
        # Mtile = MDIM // 128 // 2
        # Ntile = NDIM // 512
        # Ktile = KDIM // 128

        # # -- Single kernel call for both gate & up
        # big_output_padded = nki_matmul_fully_optimized_spmd_proj_fused[
        #     nl.nc(2)
        # ](
        #     x_flat_padded_T,    # shape (C, M_pad) => (K, M)
        #     combined_weight,  # shape (C, out1+out2) => (K, N)
        #     Mtile,
        #     Ntile,
        #     Ktile,
        #     2    # spmd_m=2
        # )

        # # big_output_padded has shape (M_pad, out1+out2)
        # # Slice off the padding, if any
        # big_output_flat = big_output_padded[:M_orig, :]  # (M_orig, out1+out2)

        # # Split columns
        # gate_output_flat = big_output_flat[:, :out1]      # (M_orig, out1)
        # up_output_flat   = big_output_flat[:, out1:]      # (M_orig, out2)

        # # Reshape back to [B, N_, ...]
        # gate_proj_output = gate_output_flat.view(B, N_, out1)
        # up_proj_output   = up_output_flat.view(B, N_, out2)

######################################################################################################################################################

        # down_proj_input = self.act_fn(gate_proj_output) * up_proj_output

        # 1) Get their shapes
        B1, L1, E1 = gate_proj_output.shape
        B2, L2, E2 = up_proj_output.shape

        # 2) (Optional) Assert shapes match
        assert B1 == B2, f"B mismatch: {B1} != {B2}"
        assert L1 == L2, f"L mismatch: {L1} != {L2}"
        assert E1 == E2, f"E mismatch: {E1} != {E2}"

        # 3) Flatten each from 3D -> 2D
        lhs_2d = gate_proj_output.view(B1 * L1, E1)
        rhs_2d = up_proj_output.view(B2 * L2, E2)

        M_orig = B1 * L1      # Original M dimension (32)

        # We need M to be a multiple of 256 = 128 x 2core.
        BLOCK_M = 128
        M_pad = math.ceil(M_orig / BLOCK_M) * BLOCK_M  # Next multiple of 128/256

        if M_pad > M_orig:
            pad_rows = M_pad - M_orig
            pad = torch.zeros(pad_rows, E1, dtype=gate_proj_output.dtype, device=gate_proj_output.device)
            lhs_2d_padded = torch.cat([lhs_2d, pad], dim=0)
            rhs_2d_padded = torch.cat([rhs_2d, pad], dim=0)
        else:
            lhs_2d_padded = lhs_2d
            rhs_2d_padded = rhs_2d

        # 4) Run the NKI SiLU + multiply kernel on 2D
        nki_actfn_output_2d_padded = nki_silu_mul(lhs_2d_padded, rhs_2d_padded)
        if M_pad > M_orig:
            nki_actfn_output_2d = nki_actfn_output_2d_padded[:M_orig, :]
        else:
            nki_actfn_output_2d = nki_actfn_output_2d_padded

        # 5) Un-flatten back to 3D
        down_proj_input = nki_actfn_output_2d.view(B1, L1, E1)

################################################################################################################################

        output = (
            self.down_proj(down_proj_input)
            if not is_lora_module(self.up_proj)
            else self.down_proj(down_proj_input, adapter_ids)
        )
        return output

    def forward(self, x, rmsnorm=None, residual=None, adapter_ids=None):
        """
        If residual is passed in, will fuse its add into the MLP kernel

        Returns a tuple of (output, residual), where residual is the output of the residual add
        """
        if self.mlp_kernel_enabled:
            fused_rmsnorm = not self.sequence_parallel_enabled
            # Quantized MLP kernel
            if self.quantized_mlp_kernel_enabled:
                return self._kernel_enabled_quantized_mlp(
                    x, fused_rmsnorm, rmsnorm, residual, adapter_ids=adapter_ids
                )

            # TODO: Entrance to Optimization
            # MLP kernel
            return self._kernel_enabled_mlp(
                x, fused_rmsnorm, rmsnorm, residual, adapter_ids=adapter_ids
            )
            # return (self._native_mlp(x, rmsnorm, adapter_ids=adapter_ids), None)
        else:
            # No kernel

            # FY: returns a tupel(3-dim tensor, "None")
            return (self._native_mlp(x, rmsnorm, adapter_ids=adapter_ids), None)


# @register_module("NeuronLlamaAttention")
class NeuronLlamaAttention(NeuronAttentionBase):
    """
    Compared with LlamaAttention, this class just
    1. replaces the q_proj, k_proj, v_proj with column parallel layer
    2. replaces the o_proj with row parallel layer
    3. update self.num_head to be self.num_head / tp_degree
    4. update self.num_key_value_heads to be self.num_key_value_heads / tp_degree
    5. update forward() method to adjust to changes from self.num_head
    """

    def __init__(self, config: InferenceConfig, tensor_model_parallel_group=None):
        super().__init__(tensor_model_parallel_group=tensor_model_parallel_group)
        # super().__init__(config, tensor_model_parallel_group)
        self.config = config
        self.neuron_config = config.neuron_config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.padding_side = config.neuron_config.padding_side
        self.torch_dtype = config.neuron_config.torch_dtype
        self.is_medusa = config.neuron_config.is_medusa
        self.flash_decoding_enabled = config.neuron_config.flash_decoding_enabled
        self.num_cores_per_group = config.num_cores_per_group
        self.bias = getattr(config, "attention_bias", False)
        self.rpl_reduce_dtype = config.neuron_config.rpl_reduce_dtype
        self.mlp_kernel_enabled = config.neuron_config.mlp_kernel_enabled
        self.rms_norm_eps = config.rms_norm_eps

        if parallel_state.model_parallel_is_initialized():
            self.tp_degree = self.config.neuron_config.tp_degree
        else:
            self.tp_degree = 1

        self.fused_qkv = config.neuron_config.fused_qkv
        self.clip_qkv = None

        self.sequence_parallel_enabled = self.neuron_config.sequence_parallel_enabled
        self.sequence_dimension = 1 if self.sequence_parallel_enabled else None
        # logger.debug(
        #     f"Hello from NeuronLlamaAttention init! Is SP enabled? {self.sequence_parallel_enabled}. Dim? {self.sequence_dimension}"
        # )

        self.init_gqa_properties()

        self.init_rope()

    def init_rope(self):
        if not hasattr(self.config, "rope_scaling") or self.config.rope_scaling is None:
            # TODO(yihsian): Check if we can just use our own implementation
            if self.is_medusa:
                self.rotary_emb = LlamaRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.rope_theta,
                )
            else:
                self.rotary_emb = RotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.rope_theta,
                )
        else:
            rope_type = self.config.rope_scaling.get(
                "rope_type", self.config.rope_scaling.get("type", None)
            )
            if rope_type == "llama3":
                self.rotary_emb = Llama3RotaryEmbedding(
                    dim=self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.rope_theta,
                    factor=self.config.rope_scaling["factor"],
                    low_freq_factor=self.config.rope_scaling["low_freq_factor"],
                    high_freq_factor=self.config.rope_scaling["high_freq_factor"],
                    original_max_position_embeddings=self.config.rope_scaling[
                        "original_max_position_embeddings"
                    ],
                )
            else:
                # LlamaRotaryEmbedding automatically chooses the correct scaling type from config.
                # Warning: The HF implementation may have precision issues when run on Neuron.
                # We include it here for compatibility with other scaling types.
                self.rotary_emb = LlamaRotaryEmbedding(self.config)  

    def scaled_qk(self, Q, K, attention_mask):
        ## 可以操作
        ## [32,64] [64,32]
        bs, head, sequence, dimension = Q.size()
        _, head_k, sequence_k, dimension_k = K.size()
        # print("size:", sequence)
        # print("size2:", sequence_k)
        result = torch.zeros((bs, head, sequence, sequence_k), device = Q.device, dtype=Q.dtype)
        # print("Q.type:", Q.dtype)
        for i in range(bs):
            for j in range(head):
                temp_q = Q[i, j, :, :]
                temp_k = (K.transpose(2, 3))[i, j, :, :]
                # print("temp_q", temp_q.size())
                # print("temp_k", temp_k.size())
                temp = block_matmul(temp_q, temp_k)
                result[i, j, :, :] = temp
        QK = result / math.sqrt(self.head_dim)

        # QK = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.head_dim)
        QK = torch.where(attention_mask, QK, torch.finfo(QK.dtype).min)
        return QK   

# TODO: Modularize RotaryEmbedding. See how HF transformers does it in 4.43.
class Llama3RotaryEmbedding(nn.Module):
    """
    Adapted from Llama 4.43 impl
    * https://github.com/huggingface/transformers/blob/v4.43.4/src/transformers/models/llama/modeling_llama.py#L78
    * https://github.com/huggingface/transformers/blob/v4.43.4/src/transformers/modeling_rope_utils.py#L345

    This implementation ensures inv_freq is calculated and stored in fp32.
    """

    def __init__(
        self,
        dim,
        max_position_embeddings=131072,
        base=500000.0,
        factor=8.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position_embeddings=8192,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.factor = factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.old_context_len = original_max_position_embeddings
        self.register_buffer("inv_freq", None, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]

        # FY: rotary embedding latency profiling

        if self.inv_freq is None:
            inv_freq = 1.0 / (
                self.base
                ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
            )

            low_freq_wavelen = self.old_context_len / self.low_freq_factor
            high_freq_wavelen = self.old_context_len / self.high_freq_factor
            new_freqs = []
            for freq in inv_freq:
                wavelen = 2 * math.pi / freq
                if wavelen < high_freq_wavelen:
                    new_freqs.append(freq)
                elif wavelen > low_freq_wavelen:
                    new_freqs.append(freq / self.factor)
                else:
                    assert low_freq_wavelen != high_freq_wavelen
                    smooth = (self.old_context_len / wavelen - self.low_freq_factor) / (
                        self.high_freq_factor - self.low_freq_factor
                    )
                    new_freqs.append((1 - smooth) * freq / self.factor + smooth * freq)
            self.inv_freq = torch.tensor(new_freqs, dtype=inv_freq.dtype, device=inv_freq.device)

        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class NeuronLlamaDecoderLayer(nn.Module):
    """
    Just replace the attention with the NXD version, and MLP with the NXD version
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        # self.self_attn = _LLAMA_MODULE_MAP[config.neuron_config.attn_cls](
        #     config=config, tensor_model_parallel_group=get_tp_group(config)
        # )
        self.self_attn = NeuronLlamaAttention(
            config=config, tensor_model_parallel_group=get_tp_group(config)
        )
        self.mlp = NeuronLlamaMLP(config)
        # logger.debug(
        #     f"Instantiating RMSNorm modules with hidden size {config.hidden_size} and EPS {config.rms_norm_eps}"
        # )
        self.input_layernorm = None
        if (
            not config.neuron_config.is_eagle_draft
            or config.neuron_config.enable_eagle_draft_input_norm
        ):
            self.input_layernorm = get_rmsnorm_cls()(
                config.hidden_size,
                eps=config.rms_norm_eps,
                nki_enabled=config.neuron_config.nki_enabled,
            )
        self.post_attention_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
            nki_enabled=config.neuron_config.nki_enabled,
        )
        self.qkv_kernel_enabled = config.neuron_config.qkv_kernel_enabled
        self.mlp_kernel_enabled = config.neuron_config.mlp_kernel_enabled
        self.rmsnorm_quantize_kernel_enabled = config.neuron_config.rmsnorm_quantize_kernel_enabled
        self.mlp_kernel_fuse_residual_add = config.neuron_config.mlp_kernel_fuse_residual_add
        self.sequence_parallel_enabled = config.neuron_config.sequence_parallel_enabled
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        adapter_ids=None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states


        # if SIMPLE_PROFILE:
        #     rmsnorm_self_attn_start = time.time()
        
        # RMSNorm (fused with QKV kernel when SP is disabled)
        if (not self.qkv_kernel_enabled or self.sequence_parallel_enabled) and self.input_layernorm:
            hidden_states = self.input_layernorm(hidden_states)

        # if SIMPLE_PROFILE:
        #     rmsnorm_self_attn_end = time.time()
        #     print(f"Layer {self.layer_index}: rms norm in Self-attention time = {rmsnorm_self_attn_end - rmsnorm_self_attn_start:.6f} s")

        # Self Attention

        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            adapter_ids=adapter_ids,
            rmsnorm=self.input_layernorm,
            **kwargs,
        )



        if self.mlp_kernel_enabled and self.mlp_kernel_fuse_residual_add:
            assert (
                not self.sequence_parallel_enabled
            ), "mlp_kernel_fuse_residual_add should be off when sequence parallelism is enabled"
            # First residual add handled in the MLP kernel


            hidden_states, residual = self.mlp(
                hidden_states,
                rmsnorm=self.post_attention_layernorm,
                residual=residual,
                adapter_ids=adapter_ids,
            )

        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            # RMSNorm (fused with QKV kernel when SP is disabled)
            if not self.mlp_kernel_enabled or self.sequence_parallel_enabled:
                hidden_states = self.post_attention_layernorm(hidden_states)

            hidden_states, _ = self.mlp(
                hidden_states,
                rmsnorm=self.post_attention_layernorm,
                adapter_ids=adapter_ids,
            )

        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache)
        return outputs


class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))


class NeuronLlamaModel(NeuronBaseModel):
    """
    The neuron version of the LlamaModel
    """

    def setup_attr_for_model(self, config: InferenceConfig):
        # Needed for init_inference_optimization()
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: InferenceConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if parallel_state.model_parallel_is_initialized():
            self.embed_tokens = ParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
                dtype=config.neuron_config.torch_dtype,
                shard_across_embedding=not config.neuron_config.vocab_parallel,
                sequence_parallel_enabled=False,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
                use_spmd_rank=config.neuron_config.vocab_parallel,
            )


            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                gather_output=not self.on_device_sampling,
                bias=False,
                pad=True,
                tensor_model_parallel_group=get_tp_group(config),
            )
        else:
            self.embed_tokens = nn.Embedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
            )
            self.lm_head = nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
            )

        # In the target fp8 checkpoint, the 1st and last
        # layers are not using fp8.
        updated_configs = []
        for i in range(config.num_hidden_layers):
            # TODO: Remove hardcoded code to have non-quantized MLPs for first and last decoder block
            if i == 0 or i == config.num_hidden_layers - 1:
                non_quant_config = copy.deepcopy(config)
                non_quant_config.neuron_config.quantized_mlp_kernel_enabled = False
                updated_configs.append(non_quant_config)
            else:
                updated_configs.append(config)
        self.layers = nn.ModuleList([NeuronLlamaDecoderLayer(conf) for conf in updated_configs])

        # FY: mark each layer with numerical value
        for i, layer in enumerate(self.layers):
            layer.layer_index = i

        if not config.neuron_config.is_eagle_draft:
            self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps, nki_enabled=config.neuron_config.nki_enabled)

        if config.neuron_config.is_eagle_draft:
            fc_bias = getattr(config, "fc_bias", False)
            self.fc = ColumnParallelLinear(
                config.hidden_size * 2, config.hidden_size, bias=fc_bias, gather_output=True
            )
        self.is_medusa = config.neuron_config.is_medusa
        self.num_medusa_heads = config.neuron_config.num_medusa_heads
        self.medusa_speculation_length = config.neuron_config.medusa_speculation_length

        if self.is_medusa:
            if parallel_state.model_parallel_is_initialized():
                medusa_head_cls = ColumnParallelLinear
            else:
                medusa_head_cls = nn.Linear
            for i in range(self.num_medusa_heads):
                medusa_head = nn.Sequential(
                    *([ResBlock(config.hidden_size)] * 1),
                    medusa_head_cls(
                        config.hidden_size,
                        config.vocab_size,
                        gather_output=not self.on_device_sampling,
                        bias=False,
                    ),
                )
                setattr(self, f"medusa_head_{i}", medusa_head)


class NeuronLlamaForCausalLM(NeuronBaseForCausalLM):
    """
    This class extends LlamaForCausalLM create traceable
    blocks for Neuron.

    Args:
        LlamaForCausalLM (_type_): _description_
    """

    _model_cls = NeuronLlamaModel

    @staticmethod
    def load_hf_model(model_path):
        return LlamaForCausalLM.from_pretrained(model_path)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        """This function should be over-ridden in child classes as needed"""
        neuron_config = config.neuron_config
        if neuron_config.fused_qkv:
            state_dict = convert_state_dict_to_fused_qkv(state_dict, config)

        if neuron_config.vocab_parallel:
            # TODO: this hack can be removed after replication_id is ready to use
            state_dict["embed_tokens.rank_util.rank"] = torch.arange(
                0, neuron_config.local_ranks_size
            )

        # to facilitate rank usage in attention
        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        for i in range(num_layers):
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(
                0, tp_degree, dtype=torch.int32
            )
        # to facilitate rank usage in base model
        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        return state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_config_cls(cls):
        return LlamaInferenceConfig
