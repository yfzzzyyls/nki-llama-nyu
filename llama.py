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
from torch import nn, ones
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
)
from neuronx_distributed_inference.modules.attention.utils import (
    RotaryEmbedding,
    preprocess_quantized_linear_layer,
    transpose_parallel_linear_layer,
)

# from neuronx_distributed_inference.modules.custom_calls import CustomRMSNorm
from neuronx_distributed_inference.modules.flashdecode.utils import calculate_num_cores_per_group
from neuronx_distributed_inference.modules.lora_serving.lora_module import is_lora_module
from neuronx_distributed_inference.utils.distributed import get_tp_group

from torch_neuronx.xla_impl.ops import RmsNorm

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

import time

SIMPLE_PROFILE = False

_LLAMA_MODULE_MAP = {}

logger = logging.getLogger("Neuron")
logger.setLevel(level=logging.DEBUG)
handler = logging.FileHandler("./app.log", encoding='UTF-8')
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# @nki.jit
# def nki_matmul_fully_optimized_(
#     lhsT,
#     rhs,
#     TILES_IN_BLOCK_M=16,
#     TILES_IN_BLOCK_N=2,
#     TILES_IN_BLOCK_K=8,
# ):
#     return 0

@nki.jit
def nki_mat_mul_test(lhs, rhs):

    M, K = lhs.shape         # [M, K]
    K2, N = rhs.shape        # [K, N]
    assert K == K2

    TILE_M = 128
    TILE_K = 128
    TILE_N = 512
    TILES_IN_BLOCK_M = 16
    TILES_IN_BLOCK_N = 2
    TILES_IN_BLOCK_K = 8

    BLOCK_M = TILE_M * TILES_IN_BLOCK_M  # 128*16=2048
    BLOCK_N = TILE_N * TILES_IN_BLOCK_N  # 512*2=1024
    BLOCK_K = TILE_K * TILES_IN_BLOCK_K  # 128*8=1024

    result = nl.zeros((M, N), dtype=lhs.dtype, buffer=nl.shared_hbm)

    aligned = (M % BLOCK_M == 0) and (N % BLOCK_N == 0) and (K % BLOCK_K == 0)
    if aligned:
        temp_aligned = nki_matmul_fully_optimized_(lhs.T, rhs)

        i_xy = nl.mgrid[0:M, 0:N]
        nl.store(
            result[i_xy.p, i_xy.x],
            temp_aligned[i_xy.p, i_xy.x]
        )
        return result

    M_main = (M // BLOCK_M) * BLOCK_M
    N_main = (N // BLOCK_N) * BLOCK_N
    K_main = (K // BLOCK_K) * BLOCK_K

    if (M_main > 0) and (N_main > 0) and (K_main > 0):
        lhs_mainT = nl.ndarray((K_main, M_main), dtype=lhs.dtype, buffer=nl.sbuf)
        i_lhs = nl.mgrid[0:K_main, 0:M_main]
        nl.load(
            lhs[i_lhs.x, i_lhs.p], 
            out=lhs_mainT[i_lhs.p, i_lhs.x]
        )

        rhs_main = nl.ndarray((K_main, N_main), dtype=rhs.dtype, buffer=nl.sbuf)
        i_rhs = nl.mgrid[0:K_main, 0:N_main]
        nl.load(
            rhs[i_rhs.p, i_rhs.x],
            out=rhs_main[i_rhs.p, i_rhs.x]
        )

        out_main = nki_matmul_fully_optimized_(lhs_mainT, rhs_main)

        i_main = nl.mgrid[0:M_main, 0:N_main]
        nl.store(
            result[i_main.p, i_main.x],
            out_main[i_main.p, i_main.x]
        )

    K_tail = K - K_main
    if (M_main > 0) and (N_main > 0) and (K_tail > 0):
        res_sub = nl.copy(
            result[0:M_main, 0:N_main], 
            dtype=nl.float32,           
            buffer=nl.psum
        )

        i_sub = nl.mgrid[0:M_main, 0:N_main]
        for kk in nl.sequential_range(K_tail):
            lhs_col = nl.load(lhs[i_sub.p, K_main + kk])     
            rhs_row = nl.load(rhs[K_main + kk, i_sub.x])        
            res_sub[i_sub.p, i_sub.x] += lhs_col[i_sub.p] * rhs_row[i_sub.x]

        nl.store(
            result[i_sub.p, i_sub.x],
            res_sub[i_sub.p, i_sub.x]
        )

    N_tail = N - N_main
    if (M_main > 0) and (N_tail > 0):
        i_tr = nl.mgrid[0:M_main, 0:N_tail] 
        res_tr = nl.zeros((M_main, N_tail), dtype=nl.float32, buffer=nl.psum)
        for kk in nl.sequential_range(K): 
            lhs_col = nl.load(lhs[i_tr.p, kk])     
            rhs_col = nl.load(rhs[kk, N_main + i_tr.x]) 
            res_tr[i_tr.p, i_tr.x] += lhs_col[i_tr.p] * rhs_col[i_tr.x]

        nl.store(
            result[i_tr.p, N_main + i_tr.x],
            res_tr[i_tr.p, i_tr.x]
        )

    M_tail = M - M_main
    if (M_tail > 0) and (N_main > 0):
        i_bl = nl.mgrid[0:M_tail, 0:N_main]
        res_bl = nl.zeros((M_tail, N_main), dtype=nl.float32, buffer=nl.psum)
        for kk in nl.sequential_range(K):
            lhs_col = nl.load(lhs[M_main + i_bl.p, kk])    
            rhs_col = nl.load(rhs[kk, i_bl.x])              
            res_bl[i_bl.p, i_bl.x] += lhs_col[i_bl.p] * rhs_col[i_bl.x]

        nl.store(
            result[M_main + i_bl.p, i_bl.x],
            res_bl[i_bl.p, i_bl.x]
        )

    if (M_tail > 0) and (N_tail > 0):
        i_br = nl.mgrid[0:M_tail, 0:N_tail]
        res_br = nl.zeros((M_tail, N_tail), dtype=nl.float32, buffer=nl.psum)
        for kk in nl.sequential_range(K):
            lhs_col = nl.load(lhs[M_main + i_br.p, kk])        # shape [M_tail]
            rhs_col = nl.load(rhs[kk, N_main + i_br.x])        # shape [N_tail]
            res_br[i_br.p, i_br.x] += lhs_col[i_br.p] * rhs_col[i_br.x]

        nl.store(
            result[M_main + i_br.p, N_main + i_br.x],
            res_br[i_br.p, i_br.x]
        )

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
def nki_matmul_quantized_fp8_rhs(
    lhsT,
    rhs_fp8,
    # Meta-parameters
    TILES_IN_BLOCK_M=16,
    TILES_IN_BLOCK_N=2,
    TILES_IN_BLOCK_K=8,
):

    # Dimensions
    K, M = lhsT.shape        # lhsT is shape [K, M]
    K_, N = rhs_fp8.shape    # rhs_fp8 is shape [K, N]
    assert K == K_, "lhsT and rhs_fp8 must have the same K dimension"

    # Prepare the result in BF16
    result = nl.ndarray((M, N), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    # Tiling sizes derived from hardware constraints
    TILE_M = nl.tile_size.gemm_stationary_fmax  # Typically 128
    TILE_K = nl.tile_size.pmax                  # Typically 128
    TILE_N = nl.tile_size.gemm_moving_fmax      # Typically 512

    BLOCK_M = TILE_M * TILES_IN_BLOCK_M
    BLOCK_N = TILE_N * TILES_IN_BLOCK_N
    BLOCK_K = TILE_K * TILES_IN_BLOCK_K

    # Validate that M, N, K are multiples of block sizes
    assert M % BLOCK_M == 0, f"M={M} not multiple of BLOCK_M={BLOCK_M}"
    assert N % BLOCK_N == 0, f"N={N} not multiple of BLOCK_N={BLOCK_N}"
    assert K % BLOCK_K == 0, f"K={K} not multiple of BLOCK_K={BLOCK_K}"

    NUM_BLOCK_M = M // BLOCK_M
    NUM_BLOCK_N = N // BLOCK_N
    NUM_BLOCK_K = K // BLOCK_K

    # -------------------------------------------------------------------------
    # 1) Load the entire FP8 rhs into on-chip memory (SBUF) just once
    # -------------------------------------------------------------------------
    # ***IMPORTANT***: This requires that K*N fits in on-chip memory, which
    # may be very large. If rhs is too big, you need a more advanced tiling approach.
    rhs_sbuf = nl.ndarray((K, N), dtype=nl.float8_e4m3, buffer=nl.sbuf)
    rhs_sbuf[...] = nl.load(rhs_fp8)  # single load from HBM -> SBUF
    # Now 'rhs_sbuf' holds the entire rhs in FP8 E4M3 format on-chip.

    # -------------------------------------------------------------------------
    # 2) Matrix Multiply: For each block of M, N, K
    # -------------------------------------------------------------------------
    for n in nl.affine_range(NUM_BLOCK_N):
        # result_tiles shape => (NUM_BLOCK_M, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILE_M, TILE_N)
        # We'll accumulate partial results in BF16 (instead of float32).
        result_tiles = nl.zeros((NUM_BLOCK_M, TILES_IN_BLOCK_M, TILES_IN_BLOCK_N,
                                 nl.par_dim(TILE_M), TILE_N),
                                dtype=nl.bfloat16,  # partial sums in BF16
                                buffer=nl.sbuf)

        # ---------------------------------------------------------------------
        # 2a) Block over K dimension
        # ---------------------------------------------------------------------
        for k_blk in nl.sequential_range(NUM_BLOCK_K):
            # We'll load sub-tiles of 'lhsT' from HBM repeatedly.
            # But for rhs, we have everything in rhs_sbuf. We just index from it.

            # Prepare a SBUF buffer for sub-tiles of shape (TILES_IN_BLOCK_K, TILE_K, BLOCK_N).
            i_rhs = nl.mgrid[0:TILE_K, 0:BLOCK_N]
            rhs_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_N),
                                   dtype=nl.bfloat16,  # We'll cast FP8->BF16
                                   buffer=nl.sbuf)

            # Load sub-blocks of 'rhs_sbuf' from on-chip memory (FP8) and cast to BF16
            for bk_r in nl.affine_range(TILES_IN_BLOCK_K):
                # Indices for the K dimension
                k_offset = (TILES_IN_BLOCK_K * k_blk + bk_r) * TILE_K
                rhs_tiles[bk_r, i_rhs.p, i_rhs.x] = nl.cast(
                    rhs_sbuf[k_offset + i_rhs.p, BLOCK_N * n + i_rhs.x],
                    dtype=nl.bfloat16
                )

            # -----------------------------------------------------------------
            # 2b) Block over M dimension
            # -----------------------------------------------------------------
            for m in nl.affine_range(NUM_BLOCK_M):
                # Load sub-blocks of lhsT (still in BF16) from HBM
                i_lhsT = nl.mgrid[0:TILE_K, 0:BLOCK_M]
                lhsT_tiles = nl.ndarray((TILES_IN_BLOCK_K, nl.par_dim(TILE_K), BLOCK_M),
                                        dtype=lhsT.dtype,  # BF16
                                        buffer=nl.sbuf)
                for bk_l in nl.affine_range(TILES_IN_BLOCK_K):
                    lhsT_tiles[bk_l, i_lhsT.p, i_lhsT.x] = nl.load(
                        lhsT[(TILES_IN_BLOCK_K * k_blk + bk_l) * TILE_K + i_lhsT.p,
                             BLOCK_M * m + i_lhsT.x])

                # -------------------------------------------------------------
                # 2c) Perform the local matmul across sub-tiles
                # -------------------------------------------------------------
                i_lhsT_mm = nl.mgrid[0:TILE_K, 0:TILE_M]   # shape => [TILE_K, TILE_M]
                i_rhs_mm = nl.mgrid[0:TILE_K, 0:TILE_N]    # shape => [TILE_K, TILE_N]
                i_res_mm = nl.mgrid[0:TILE_M, 0:TILE_N]    # shape => [TILE_M, TILE_N]

                for bn in nl.affine_range(TILES_IN_BLOCK_N):
                    for bm in nl.affine_range(TILES_IN_BLOCK_M):
                        # BF16 partial sum tile
                        res_tile = nl.zeros((TILE_M, TILE_N), dtype=nl.bfloat16, buffer=nl.psum)

                        # Loop over TILES_IN_BLOCK_K to accumulate partial products
                        for bk in nl.affine_range(TILES_IN_BLOCK_K):
                            # Multiply LHS (BF16) with RHS (BF16).
                            # nisa.nc_matmul automatically promotes to higher precision internally
                            # then accumulates into BF16 partial sum.
                            lhs_sub = lhsT_tiles[bk, i_lhsT_mm.p, bm * TILE_M + i_lhsT_mm.x]
                            rhs_sub = rhs_tiles[bk, i_rhs_mm.p, bn * TILE_N + i_rhs_mm.x]
                            res_tile[...] += nisa.nc_matmul(lhs_sub, rhs_sub)

                        # Add partial sums to result_tiles in BF16
                        result_tiles[m, bm, bn, i_res_mm.p, i_res_mm.x] += res_tile[
                            i_res_mm.p, i_res_mm.x
                        ]

        # ---------------------------------------------------------------------
        # 2d) Copy partial results from SBUF to final result in HBM
        # ---------------------------------------------------------------------
        for m in nl.affine_range(NUM_BLOCK_M):
            for bm in nl.affine_range(TILES_IN_BLOCK_M):
                i_res = nl.mgrid[0:TILE_K, 0:TILE_N]
                i_res_packed = nl.mgrid[0:TILE_K, 0:BLOCK_N]
                result_packed = nl.ndarray((TILE_K, BLOCK_N),
                                           dtype=result_tiles.dtype,
                                           buffer=nl.sbuf)
                # Coalesce tiles for better DMA
                for bn in nl.affine_range(TILES_IN_BLOCK_N):
                    result_packed[i_res.p, bn * TILE_N + i_res.x] = nl.copy(
                        result_tiles[m, bm, bn, i_res.p, i_res.x]
                    )
                # Finally store to 'result' in BF16
                nl.store(result[(TILES_IN_BLOCK_M * m + bm) * TILE_K + i_res_packed.p,
                                BLOCK_N * n + i_res_packed.x],
                         value=result_packed[i_res_packed.p, i_res_packed.x])

    return result

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
def nki_matmul_tiled_(lhsT, rhs):
    """NKI kernel to compute a matrix multiplication operation in a tiled manner

    Args:
        lhsT: an input tensor of shape [K,M], where both K and M are multiples for
            128.  It is the left-hand-side argument of the matrix multiplication,
            delivered transposed for optimal performance.
        rhs: an input tensor of shape [K,N], where K is a multiple of 128, and N
            is a multiple of 512.  It is the right-hand-side argument of the matrix
            multiplication.
    Returns:
        result: the resulting output tensor of shape [M,N]
    """

    K, M = lhsT.shape
    print(K, M)
    K_, N = rhs.shape
    print(K_, N)
    assert K == K_, "lhsT and rhs must have the same contraction dimension"
    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    # TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    TILE_M = M
    TILE_K = nl.tile_size.pmax  # 128
    TILE_N = nl.tile_size.gemm_moving_fmax  # 512

    # Use affine_range to loop over tiles
    for m in nl.affine_range(M // TILE_M):
        for n in nl.affine_range(N // TILE_N):
            # Allocate a tensor in PSUM
            res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)

            for k in nl.affine_range(K // TILE_K):
                # Declare the tiles on SBUF
                lhsT_tile = nl.ndarray((TILE_K, TILE_M), dtype=lhsT.dtype, buffer=nl.sbuf)
                rhs_tile = nl.ndarray((TILE_K, TILE_N), dtype=rhs.dtype, buffer=nl.sbuf)

                # Load tiles from lhsT and rhs
                lhsT_tile[...] = nl.load(lhsT[k * TILE_K:(k + 1) * TILE_K,
                                            m * TILE_M:(m + 1) * TILE_M])
                rhs_tile[...] = nl.load(rhs[k * TILE_K:(k + 1) * TILE_K,
                                            n * TILE_N:(n + 1) * TILE_N])

                # Accumulate partial-sums into PSUM
                res_psum += nl.matmul(lhsT_tile[...], rhs_tile[...], transpose_x=True)

            # Copy the result from PSUM back to SBUF, and cast to expected output data-type
            res_sb = nl.copy(res_psum, dtype=result.dtype)
            nl.store(result[m * TILE_M:(m + 1) * TILE_M, n * TILE_N:(n + 1) * TILE_N],
                value=res_sb)
    logger.debug("nki函数被调用")
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
def nki_rmsnorm_kernel(a_tensor, g_tensor, eps):
    # Calculate out_tensor = a_tensor/RMS(a_tensor) * g_tensor
    # Where RMS(a_tensor) = sqrt((1/N) * sum(a_tensor * a_tensor))
    # and N = a_tensor.shape[1]
    # Reduction (mean) is performed in the free (2nd) dimension




    if __debug__ and SIMPLE_PROFILE:
        rms_kernel_start = time.time()

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
            print("a tensor shape2", a_tensor.shape[2])
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

    if __debug__ and SIMPLE_PROFILE:
        rms_kernel_end = time.time()
        print(f"RMSNorm Kernel time: {rms_kernel_end - rms_kernel_start:.6f} s")


    return out_tensor


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
        if self.nki_enabled:
            out_tensor = nki_rmsnorm_kernel(hidden_states, self.weight, self.variance_epsilon)
            return out_tensor

        original_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
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

        if __debug__ and SIMPLE_PROFILE:
            self.total_dup_time = 0.0  
            self.total_wgate_time = 0.0  
            self.total_wup_time = 0.0  
            self.total_act_time = 0.0  
            self.total_mul_time = 0.0  
            self.total_wdown_time = 0.0  
            self.total_fused_mlp_time = 0.0  
            self.call_count = 0  


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


        # Print all the parameters:
        print("hidden_size:", self.hidden_size)
        print("intermediate_size:", self.intermediate_size)
        print("bias:", mlp_bias)
        print("dtype:", config.neuron_config.torch_dtype)
        print("tensor_model_parallel_group:", get_tp_group(config))
        print("多层感知机内核使能状态:", self.mlp_kernel_enabled)
        print("多层感知机NKI使能状态", self.neuron_config.nki_enabled)
        print("量化多层感知机内核使能状态", self.quantized_mlp_kernel_enabled)
        print("硬件并行状态", parallel_state.model_parallel_is_initialized())

        if parallel_state.model_parallel_is_initialized():
            if self.quantized_mlp_kernel_enabled:
                # Quantized MLP kernels expect intermediate size to be multiple of 128, so we need to pad
                tp_degree = self.neuron_config.tp_degree
                self.intermediate_size += (
                    get_padding_length(self.intermediate_size // tp_degree, 128) * tp_degree
                )
                logger.debug(f"Quantized intermediate_size: {self.intermediate_size}")

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
                print("initialize columnxxx")
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
        logger.debug(
            f"MLP: quantized kernel, fused_residual={fused_residual}, fused_rmsnorm={fused_rmsnorm}, logical_neuron_cores={self.logical_neuron_cores}"
        )

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
                logger.debug(
                    "Running Quantized MLP kernel with sequence-parallel RMSnorm-Quantize kernel!"
                )
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
                logger.debug(
                    "Running Quantized MLP kernel with external (native compiler) sequence-parallel RMSnorm!"
                )
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

        logger.debug(f"Quantized MLP output shape {output_tensor.shape}")
        return (output_tensor, residual)

    def _kernel_enabled_mlp(self, x, fused_rmsnorm, rmsnorm, residual, adapter_ids):
        fused_residual = residual is not None
        logger.debug(
            f"MLP: kernel, fused_residual={fused_residual}, fused_rmsnorm={fused_rmsnorm}, logical_neuron_cores={self.logical_neuron_cores}"
        )

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

        logger.debug(f"MLP output shape {output_tensor.shape}")
        return (output_tensor, residual)

    def _native_mlp(self, x, rmsnorm, adapter_ids=None):
        logger.debug("MLP: native compiler")
        # all-gather is done here instead of CPL layers to
        # avoid 2 all-gathers from up and gate projections

        # print("use native MLP")
        # print("use FFN NKI kernel implementation")

        # print("Input dimensions:", x.shape)

        if __debug__ and SIMPLE_PROFILE:
            dup_start = time.time()

        if self.sequence_parallel_enabled:
            x = gather_from_sequence_parallel_region(
                x, self.sequence_dimension, process_group=get_tp_group(self.config)
            )
        if __debug__ and SIMPLE_PROFILE:
            self.dup_time = time.time() - dup_start

        if __debug__ and SIMPLE_PROFILE: wgate_start = time.time()



        # FFN Kernel gate project replacement starts here ################################################

        # default size in evaluation mode
        # input 1 x 32 x 2048 or 1 x 1 x 2048
        # hidden : 2048
        # intermediate: 8192

        # extract weight Wgate
        # Wgate = self.gate_proj.weight.detach().clone()
        Wgate = self.gate_proj.weight.data
        Wup = self.gate_proj.weight.data
        Wdown = self.down_proj.weight.data

        # information printout
        logger.debug(f"门网络权重形状：{Wgate.shape}; 类型：{type(Wgate)}")
        logger.debug(f"上网络权重形状：{Wup.shape}")
        logger.debug(f"下网络权重形状：{Wdown.shape}")
        # print("gate weight data", Wgate)
        # print("input shape", x.shape[1], x.shape[2])
        logger.debug(f"输入数据形状：{x.shape}")
        # print("batch 0 input data", x[0])

        # transpose input x, x is dimenson 1 x 32 x 2048: use the bottom python transpose function
        # tensor_transpose2D_kernel_(x[0], (x.shape[1], x.shape[2]))
        x_transposed = x.transpose(1, 2)

        # do nki matmul: use the tiled version
        batch_size, sequence_length, dimension = x.shape
        output_dimension = max(Wgate.shape)
        nki_gate_proj_output = torch.zeros((batch_size, sequence_length, output_dimension), dtype=x.dtype, device=x.device)
        for i in range(batch_size):
            mul_result = nki_matmul_tiled_(x_transposed[i], Wgate.T)
            print(f"Multiplication Result shape: {mul_result.shape}")
            nki_gate_proj_output[i] = mul_result
        logger.debug(f"门网络输出形状：{nki_gate_proj_output.shape}")
        # gate_proj_output = nki_matmul_fully_optimized_(x_transposed[0], Wgate, 1, 16, 16)
        # gate_proj_output = nki_matmul_basic_(x_transposed[0], Wgate)
        
        # gate_proj_kernel_output = nki_matmul_tiled_(x_transposed[0], Wgate)
        # # gate_proj_output = (gate_proj_kernel_output, None)

        # print("Kernel Result: ", gate_proj_kernel_output)

        # gate_proj_output = self.gate_proj(x)
        # print("Gold Result: ", gate_proj_output)

        gate_proj_output = (
            nki_gate_proj_output
            if not is_lora_module(self.gate_proj)
            else self.gate_proj(x, adapter_ids)
        )
        # below is the original implementation
        # gate_proj_output = (
        #     self.gate_proj(x)
        #     if not is_lora_module(self.gate_proj)
        #     else self.gate_proj(x, adapter_ids)
        # )
        # FFN Kernel gate project replacement ends here    ################################################

        if __debug__ and SIMPLE_PROFILE: self.wgate_time = time.time() - wgate_start

        if __debug__ and SIMPLE_PROFILE: wup_start = time.time()
        up_proj_output = (
            self.up_proj(x) if not is_lora_module(self.up_proj) else self.up_proj(x, adapter_ids)
        )
        if __debug__ and SIMPLE_PROFILE: self.wup_time = time.time() - wup_start

        print("Check")
        print(f"Gate Shape:{gate_proj_output.shape}")
        print(f"Up Shape:{up_proj_output.shape}")
        print(f"Intermediate Size: {self.intermediate_size}")
        down_proj_input = self.act_fn(gate_proj_output) * up_proj_output
        
        if __debug__ and SIMPLE_PROFILE: wdown_start = time.time()

################################################
        # Wdown = self.down_proj.weight.detach().clone()
        # print("gate weight data", Wgate)
        # print("input shape", x.shape[1], x.shape[2])
        # print("batch 0 input data", x[0])
        # tensor_transpose2D_kernel_(x[0], (x.shape[1], x.shape[2]))
        # gate_proj_output_transposed = gate_proj_output.transpose(1, 2)
        
        # gate_proj_kernel_output = nki_matmul_tiled_(x_transposed[0], Wgate)
        # output = nki_matmul_tiled_(gate_proj_output_transposed[0], Wdown)
        # print("Kernel Result: ", gate_proj_kernel_output)

        # output = self.down_proj(down_proj_input)
############################################################

        output = (
            self.down_proj(down_proj_input)
            if not is_lora_module(self.up_proj)
            else self.down_proj(down_proj_input, adapter_ids)
        )

        if __debug__ and SIMPLE_PROFILE: self.wdown_time = time.time() - wdown_start

        if __debug__ and SIMPLE_PROFILE:
            self.call_count += 1  
            self.total_dup_time += self.dup_time  
            self.total_act_time += self.act_time
            self.total_mul_time += self.mul_time
            self.total_wgate_time += self.wgate_time
            self.total_wdown_time += self.wdown_time
            self.total_wup_time += self.wup_time
            logger.info(f"MLP layer timings (token {self.call_count}): dup={self.dup_time:.6f}s, Wgate={self.wgate_time:.6f}s, Wup={self.wup_time:.6f}s, "  
                f"activation={self.act_time:.6f}s, multiply={self.mul_time:.6f}s, Wdown={self.wdown_time:.6f}s")  
            logger.info(f"MLP layer average over {self.call_count} tokens: dup={self.total_dup_time/self.call_count:.6f}s, "  
                f"Wgate={self.total_wgate_time/self.call_count:.6f}s, Wup={self.total_wup_time/self.call_count:.6f}s, "  
                f"activation={self.total_act_time/self.call_count:.6f}s, multiply={self.total_mul_time/self.call_count:.6f}s, "  
                f"Wdown={self.total_wdown_time/self.call_count:.6f}s")  


        logger.debug(f"MLP output shape {output.shape}")
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
            # return self._kernel_enabled_mlp(
            #     x, fused_rmsnorm, rmsnorm, residual, adapter_ids=adapter_ids
            # )
            return (self._native_mlp(x, rmsnorm, adapter_ids=adapter_ids), None)
        else:
            # No kernel

            # FY: returns a tupel(3-dim tensor, "None")
            return (self._native_mlp(x, rmsnorm, adapter_ids=adapter_ids), None)


@register_module("NeuronLlamaAttention")
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
        logger.debug(
            f"Hello from NeuronLlamaAttention init! Is SP enabled? {self.sequence_parallel_enabled}. Dim? {self.sequence_dimension}"
        )

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
        if __debug__ and SIMPLE_PROFILE:
            rotary_embedding_start = time.time()

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

        if __debug__ and SIMPLE_PROFILE:
            rotary_embedding_end = time.time()
            print(f"Positional encoding(rotary embedding) time: {rotary_embedding_end - rotary_embedding_start:.6f} s")

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class NeuronLlamaDecoderLayer(nn.Module):
    """
    Just replace the attention with the NXD version, and MLP with the NXD version
    """

    def __init__(self, config: InferenceConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = _LLAMA_MODULE_MAP[config.neuron_config.attn_cls](
            config=config, tensor_model_parallel_group=get_tp_group(config)
        )
        self.mlp = NeuronLlamaMLP(config)
        logger.debug(
            f"Instantiating RMSNorm modules with hidden size {config.hidden_size} and EPS {config.rms_norm_eps}"
        )
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

        if __debug__ and SIMPLE_PROFILE:
            total_start = time.time()

        # if SIMPLE_PROFILE:
        #     rmsnorm_self_attn_start = time.time()
        
        # RMSNorm (fused with QKV kernel when SP is disabled)
        if (not self.qkv_kernel_enabled or self.sequence_parallel_enabled) and self.input_layernorm:
            hidden_states = self.input_layernorm(hidden_states)

        # if SIMPLE_PROFILE:
        #     rmsnorm_self_attn_end = time.time()
        #     print(f"Layer {self.layer_index}: rms norm in Self-attention time = {rmsnorm_self_attn_end - rmsnorm_self_attn_start:.6f} s")

        # Self Attention
        if __debug__ and SIMPLE_PROFILE:
            self_attn_start = time.time()
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            adapter_ids=adapter_ids,
            rmsnorm=self.input_layernorm,
            **kwargs,
        )
        if __debug__ and SIMPLE_PROFILE:
            self_attn_end = time.time()
            print(f"Layer {self.layer_index}: Self-attention time = {self_attn_end - self_attn_start:.6f} s")


        if self.mlp_kernel_enabled and self.mlp_kernel_fuse_residual_add:
            assert (
                not self.sequence_parallel_enabled
            ), "mlp_kernel_fuse_residual_add should be off when sequence parallelism is enabled"
            # First residual add handled in the MLP kernel

            if __debug__ and SIMPLE_PROFILE:
                fused_ffn_start = time.time()
            hidden_states, residual = self.mlp(
                hidden_states,
                rmsnorm=self.post_attention_layernorm,
                residual=residual,
                adapter_ids=adapter_ids,
            )
            if __debug__ and SIMPLE_PROFILE:
                fused_ffn_end = time.time()
                print(f"Layer {self.layer_index}: Fused FFN time = {fused_ffn_end - fused_ffn_start:.6f} s")
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            # RMSNorm (fused with QKV kernel when SP is disabled)
            if not self.mlp_kernel_enabled or self.sequence_parallel_enabled:
                hidden_states = self.post_attention_layernorm(hidden_states)
            
            if __debug__ and SIMPLE_PROFILE:
                unfused_ffn_start = time.time()
            hidden_states, _ = self.mlp(
                hidden_states,
                rmsnorm=self.post_attention_layernorm,
                adapter_ids=adapter_ids,
            )
            if __debug__ and SIMPLE_PROFILE:
                unfused_ffn_end = time.time()
                print(f"Layer {self.layer_index}: FFN time = {unfused_ffn_end - unfused_ffn_start:.6f} s")

        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache)

        if __debug__ and SIMPLE_PROFILE:
            total_end = time.time()

            total_time = total_end - total_start
            print(f"Total time for token: {total_time:.6f} s")
        
            # print(f"Embedding: {embed_time/total_time:.2%}, PosEnc: {pos_time/total_time:.2%}, "
            #     f"Self-Attn: {attn_time/total_time:.2%}, FFN: {ffn_time/total_time:.2%}, "
            #     f"Final linear: {lm_time/total_time:.2%}")

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
            if __debug__ and SIMPLE_PROFILE:
                embedding_start = time.time();
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
            if __debug__ and SIMPLE_PROFILE:
                embedding_end = time.time();
                print(f"Embedding layer time: {embedding_end - embedding_start:.6f} s")

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
