import copy
import sys
import gc
import logging
import math
import numpy as np
import time
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
from neuronx_distributed_inference.modules.attention.utils import repeat_kv, manual_softmax
from torch_neuronx.xla_impl.ops import RmsNorm

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

import os
from torch_xla.core import xla_model as xm

# os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
# os.environ["NEURON_CC_FLAGS"]= " --disable-dge "

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

    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
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
def nki_matmul(lshT, rsh, mdim, cdim, ndim):
    # output tensor
    # c_tensor = nl.ndarray((2048, 4096), dtype=nki.language.bfloat16, buffer=nl.shared_hbm)
    print("mdim cdim ndim", mdim, cdim, ndim)
    c_tensor = nl.ndarray((mdim, ndim), dtype=nki.language.bfloat16, buffer=nl.shared_hbm)
    print("c_tensor shape", c_tensor.shape)
    # i_p_a = nl.arange(cdim)[:, None] # contraction p dim
    # i_f_a = nl.arange(mdim)[None, :] # lshT: M dim

    i_p_a = nl.arange(cdim)[:, None] # contraction p dim
    i_f_a = nl.arange(mdim)[None, :] # lshT: m dim
    
    i_p_b = nl.arange(cdim)[:, None] # contraction p dim
    i_f_b = nl.arange(ndim)[None, :] # rsh: N dim
    print("ipa, ifa", i_p_a.shape, i_f_a.shape)
    print("ipb, ifb", i_p_b.shape, i_f_b.shape)
    a = nl.load(lshT[i_p_a, i_f_a])
    b = nl.load(rsh[i_p_b, i_f_b])

    c_psum = nisa.nc_matmul(a[i_p_a, i_f_a], b[i_p_b, i_f_b])

    # c_psum = nisa.nc_matmul(a[i_p_a, i_f_a], b[i_f_b, i_p_b])

    i_f_a_T = nl.arange(mdim)[:, None]  # shape (mdim, 1)
    nl.store(c_tensor[i_f_a_T, i_f_b], c_psum)
    return c_tensor
    # hidden size = 2048
    # intermediate size = 8192
    # in_tensor = torch.rand((1, 32, 2048), dtype=torch.bfloat16).to(device=device)
    # gate_weight = torch.rand((2048, 8192), dtype=torch.bfloat16).to(device=device)
    # up_weight = torch.rand((2048, 8192), dtype=torch.bfloat16).to(device=device)
    # down_weight = torch.rand((8192, 2048), dtype=torch.bfloat16).to(device=device)
    # out_tensor = nki_matmul(in_tensor, gate_weight, up_weight, down_weight)
    
def test_matmul_singletile(device = xm.xla_device()):
    # lshT
    a_tensor = torch.rand((128, 67), dtype=torch.bfloat16, device=device)
    a_tensor_T = a_tensor.T # lshT
    # rsh
    b_tensor = torch.rand((67, 512), dtype=torch.bfloat16, device=device)
    assert a_tensor_T.shape[0] == b_tensor.shape[0], "hsT and rhs must have the same contraction dimension"
    output = nki_matmul(a_tensor_T, b_tensor, a_tensor_T.shape[1], a_tensor_T.shape[0], b_tensor.shape[1])
    print(f"matmul_output_shape={output.shape}") # supposed to be 2048x4096
    return output, a_tensor, b_tensor

def check_2matrices_match(default_output, nki_output):
    cpu = torch.device('cpu')
    nki_output_cpu = nki_output.to(device=cpu)

    # Ensure both outputs are on the same device (if necessary, move to CPU)
    default_output_cpu = default_output.cpu()

    # Compare using torch.allclose with a tolerance (adjust atol/rtol as needed)
    if torch.allclose(default_output_cpu, nki_output_cpu, atol=1e-2):
        print("The results match!")
    else:
        print("There is a mismatch between the outputs.")

def get_actfn(actfn_str):
    actfn = ACT2FN[actfn_str]
    return actfn

def test_nki_silu_mul(device = xm.xla_device()):
    # Generate two random 3D tensors of shape (B, L, E)
    lhs, rhs = generate_2_matrices(device)  # e.g. (1, 32, 4096)

    B1, L1, E1 = lhs.shape
    B2, L2, E2 = rhs.shape

    # 2) (Optional) Assert shapes match
    assert B1 == B2, f"B mismatch: {B1} != {B2}"
    assert L1 == L2, f"L mismatch: {L1} != {L2}"
    assert E1 == E2, f"E mismatch: {E1} != {E2}"

    # 3) Flatten each from 3D -> 2D
    lhs_2d = lhs.view(B1 * L1, E1)
    rhs_2d = rhs.view(B2 * L2, E2)

    M_orig = B1 * L1      # Original M dimension (32)

    # We need M to be a multiple of 256 = 128 x 2core.
    BLOCK_M = 128
    M_pad = math.ceil(M_orig / BLOCK_M) * BLOCK_M  # Next multiple of 256

    if M_pad > M_orig:
        pad_rows = M_pad - M_orig
        pad = torch.zeros(pad_rows, E1, dtype=lhs.dtype, device=lhs.device)
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
    nki_actfn_output_3d = nki_actfn_output_2d.view(B1, L1, E1)

    # 6) For reference, compute the default PyTorch-based result in 3D
    default_actfn = get_actfn("silu")  # e.g. F.silu
    default_actfn_output_3d = default_actfn(lhs) * rhs

    # 7) Compare the two outputs
    check_2matrices_match(default_actfn_output_3d, nki_actfn_output_3d)

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

def generate_2_matrices(device = xm.xla_device()):
    x = torch.rand((1, 256, 4096), dtype=torch.bfloat16, device=device)
    y = torch.rand((1, 256, 4096), dtype=torch.bfloat16, device=device)
    return x, y

@nki.jit
def fused_self_attn_2mask(q_ref, k_ref, v_ref, mask_ref,
                          softmax_scale=0.125, mixed_precision=True):
    """
    Fused self-attn: out = softmax( (QK^T * scale) + mask ) * V
    mask_ref is [seq_q, seq_k], boolean (True => keep, False => -∞).
    Q,K,V => [seqlen, d_head], d_head <= 128, seqlen multiple-of-128.
    """

    kernel_dtype = q_ref.dtype
    pe_in_dt = nl.bfloat16 if mixed_precision else np.float32

    seqlen, d_head = q_ref.shape
    assert k_ref.shape == (seqlen, d_head), "K shape mismatch"
    assert v_ref.shape == (seqlen, d_head), "V shape mismatch"
    assert mask_ref.shape == (seqlen, seqlen), "mask shape mismatch"
    out_ref = nl.ndarray((seqlen, d_head), dtype=kernel_dtype, buffer=nl.shared_hbm)

    # Tiling
    tile_size = 128
    assert seqlen % tile_size == 0, "Requires seqlen multiple-of-128"
    assert d_head <= 128, "Requires d_head <= 128"

    seq_n_tiles = seqlen // tile_size

    # 1) Transpose V => shape [tile_size, seq_n_tiles, d_head]
    trans_v = nl.ndarray((nl.par_dim(tile_size), seq_n_tiles, d_head), dtype=pe_in_dt)
    for it_k in nl.affine_range(seq_n_tiles):
        ip = nl.arange(tile_size)[:, None]
        if_ = nl.arange(d_head)[None, :]
        trans_v[ip, it_k, if_] = nl.load(v_ref[it_k*tile_size + ip, if_], dtype=pe_in_dt)

    # 2) Transpose Q => shape [seq_n_tiles, d_head, tile_size], also multiply by scale
    q_local = nl.ndarray((seq_n_tiles, nl.par_dim(d_head), tile_size), dtype=pe_in_dt)
    ip_q = nl.arange(d_head)[:, None]
    if_q = nl.arange(tile_size)[None, :]
    for it_q in nl.affine_range(seq_n_tiles):
        q_local[it_q, ip_q, if_q] = nl.load_transpose2d(
            q_ref[it_q*tile_size + if_q, ip_q], dtype=pe_in_dt
        ) * softmax_scale

    # 3) Transpose K => shape [seq_n_tiles, d_head, tile_size]
    k_local = nl.ndarray((seq_n_tiles, nl.par_dim(d_head), tile_size), dtype=pe_in_dt)
    ip_k = nl.arange(d_head)[:, None]
    if_k = nl.arange(tile_size)[None, :]
    for it_k in nl.affine_range(seq_n_tiles):
        k_local[it_k, ip_k, if_k] = nl.load_transpose2d(
            k_ref[it_k*tile_size + if_k, ip_k], dtype=pe_in_dt
        )

    # 4) For each Q-tile => produce QK => apply mask => softmax => multiply by V
    for it_q in nl.affine_range(seq_n_tiles):
        # We'll store partial QK results in qk_res_buf
        qk_res_buf = nl.ndarray((nl.par_dim(tile_size), seqlen), dtype=kernel_dtype)
        neg_max_res = nl.ndarray((nl.par_dim(tile_size), seq_n_tiles), dtype=kernel_dtype)

        ip_max = nl.arange(tile_size)[:, None]
        if_max = nl.arange(seq_n_tiles)[None, :]

        # (a) QK^T for each K-tile
        for it_k_2 in nl.affine_range(seq_n_tiles):
            # partial sum buffer shape [tile_size, tile_size]
            qk_psum = nl.zeros((nl.par_dim(tile_size), tile_size),
                               dtype=np.float32, buffer=nl.psum)
            ip_qk = nl.arange(tile_size)[:, None]
            if_qk = nl.arange(tile_size)[None, :]

            # Multiply Q * K
            qk_psum[ip_qk, if_qk] += nisa.nc_matmul(
                moving=k_local[it_k_2, ip_k, if_k],
                stationary=q_local[it_q, ip_q, if_q]
            )

            # global row, col
            i_mask_r = (it_q*tile_size) + ip_qk
            i_mask_c = (it_k_2*tile_size) + if_qk

            bool_mask_val = nl.load(mask_ref[i_mask_r, i_mask_c], dtype=nl.bool)
            qk_res_buf[ip_qk, it_k_2*tile_size + if_qk] = nisa.affine_select(
                pred=bool_mask_val,
                on_true_tile=qk_psum[ip_qk, if_qk],
                on_false_value=-9984.0,  # approximate -inf
                dtype=kernel_dtype
            )

            neg_max_res[ip_max, it_k_2] = nisa.tensor_reduce(
                np.max,
                data=qk_res_buf[ip_qk, it_k_2*tile_size + if_qk],
                axis=(1,),
                dtype=kernel_dtype,
                negate=True
            )

        neg_max_res_final = nisa.tensor_reduce(
            np.min,
            data=neg_max_res[ip_max, if_max],
            axis=(1,),
            dtype=kernel_dtype,
            negate=False
        )

        # (b) exponent + sum
        ip_softmax = nl.arange(tile_size)[:, None]
        if_softmax = nl.arange(seqlen)[None, :]
        exp_res = nisa.activation(
            np.exp,
            data=qk_res_buf[ip_softmax, if_softmax],
            bias=neg_max_res_final,
            scale=1.0
        )

        sum_res = nisa.tensor_reduce(
            np.add,
            data=exp_res,
            axis=(1,),
            dtype=kernel_dtype
        )

        softmax_res = nl.ndarray((nl.par_dim(tile_size), seqlen), dtype=pe_in_dt)
        softmax_res[ip_softmax, if_softmax] = nl.copy(exp_res, dtype=pe_in_dt)

        sum_recip = 1.0 / sum_res
        sum_divisor = nl.ndarray((nl.par_dim(tile_size), tile_size), dtype=kernel_dtype)
        sum_divisor[...] = nl.copy(sum_recip.broadcast_to((tile_size, tile_size)), dtype=kernel_dtype)

        # (c) multiply by V
        trans_softmax_res = nl.ndarray((nl.par_dim(tile_size), seq_n_tiles, tile_size), dtype=pe_in_dt)
        for it_k_2 in nl.affine_range(seq_n_tiles):
            ip_scores = nl.arange(tile_size)[:, None]
            if_scores = nl.arange(tile_size)[None, :]
            trans_softmax_res[ip_scores, it_k_2, if_scores] = nisa.nc_transpose(
                softmax_res[ip_scores, it_k_2*tile_size + if_scores]
            )

        attn_res_psum = nl.zeros((nl.par_dim(d_head), tile_size), dtype=np.float32, buffer=nl.psum)
        ip_out = nl.arange(d_head)[:, None]
        if_out = nl.arange(tile_size)[None, :]

        for it_k_2 in nl.affine_range(seq_n_tiles):
            ip_vt = nl.arange(tile_size)[:, None]
            if_vt = nl.arange(d_head)[None, :]
            attn_res_psum[ip_out, if_out] += nisa.nc_matmul(
                moving=trans_softmax_res[ip_scores, it_k_2, if_scores],
                stationary=trans_v[ip_vt, it_k_2, if_vt]
            )

        attn_res_sbuf = nl.copy(attn_res_psum[ip_out, if_out], dtype=kernel_dtype)

        # multiply by sum_div
        sum_div_trans = nisa.nc_transpose(sum_divisor[:, :tile_size])
        attn_res_div = attn_res_sbuf * sum_div_trans

        # store final
        nl.store(
            out_ref[it_q*tile_size + if_out, ip_out],
            value=attn_res_div
        )

    return out_ref

def compute_for_token_gen_ref(
    Q, K, V,
    past_key_value,
    attention_mask,
    active_mask,
):
    """
    Original reference logic, without 'self'.
    1. Expand 'prior' K,V from past_key_value
    2. Expand 'active' K,V from K,V
    3. Score => softmax => multiply => sum
    """
    # 1) prior
    K_prior, V_prior = past_key_value
    # For simplicity, skip 'repeat_kv' and assume K_prior, V_prior are correct shape
    # If needed, do:
    #   K_prior = repeat_kv(K_prior, num_key_value_groups)
    #   V_prior = repeat_kv(V_prior, num_key_value_groups)

    head_dim = Q.shape[-1]
    prior_scores = torch.matmul(Q, K_prior.transpose(-2, -1)) / math.sqrt(head_dim)
    prior_scores = torch.where(attention_mask, prior_scores, torch.finfo(prior_scores.dtype).min)
    prior_scores = prior_scores.float()  # do softmax in float32

    # 2) active
    # If needed, repeat
    K_active = K
    V_active = V
    active_scores = torch.matmul(Q, K_active.transpose(-2, -1)) / math.sqrt(head_dim)
    active_scores = torch.where(active_mask, active_scores, torch.finfo(active_scores.dtype).min)
    active_scores = active_scores.float()

    # 3) manual softmax => two slices => multiply => sum
    # We'll do a simpler approach: prior_p = softmax(prior_scores), active_p = softmax(active_scores)
    # Then total attn_prior + attn_active
    prior_probs = torch.softmax(prior_scores, dim=-1).to(Q.dtype)
    active_probs = torch.softmax(active_scores, dim=-1).to(Q.dtype)

    attn_prior = torch.matmul(prior_probs, V_prior)
    attn_active = torch.matmul(active_probs, V_active)
    attn_output = attn_prior + attn_active
    return attn_output

def compute_for_token_gen_nki(
    Q, K, V,
    past_key_value,
    attention_mask,
    active_mask,
):
    """
    Uses the fused_self_attn_2mask kernel twice (prior vs. active),
    then sums the results.
    Q, K, V shape: [B, heads, seqQ, d_head]
    attention_mask shape: [B, heads, seqQ, seqPrior]
    """
    B, H, seqQ, d_head = Q.shape
    # 1) Flatten Q -> shape [B*H*seqQ, d_head]
    Q_2d = Q.reshape(-1, d_head)
    # Build prior K, V
    K_prior, V_prior = past_key_value
    seqPrior = K_prior.shape[2]
    Kp_2d = K_prior.reshape(B*H*seqPrior, d_head)
    Vp_2d = V_prior.reshape(B*H*seqPrior, d_head)

    # Flatten attention_mask -> shape [B*H*seqQ, B*H*seqPrior]
    attn_mask_2d = attention_mask.reshape(B*H*seqQ, B*H*seqPrior)

    # 2) fused kernel => attn_prior
    scale = 1.0 / math.sqrt(d_head)
    attn_prior_2d = fused_self_attn_2mask(Q_2d, Kp_2d, Vp_2d, attn_mask_2d,
                                          softmax_scale=scale, mixed_precision=True)
    # reshape back
    attn_prior = attn_prior_2d.view(B, H, seqQ, d_head)

    # 3) "active" K, V
    seqActive = K.shape[2]
    Ka_2d = K.reshape(B*H*seqActive, d_head)
    Va_2d = V.reshape(B*H*seqActive, d_head)
    active_mask_2d = active_mask.reshape(B*H*seqQ, B*H*seqActive)

    attn_active_2d = fused_self_attn_2mask(Q_2d, Ka_2d, Va_2d, active_mask_2d,
                                           softmax_scale=scale, mixed_precision=True)
    attn_active = attn_active_2d.view(B, H, seqQ, d_head)

    # 4) Sum and return
    return attn_prior + attn_active

def test_self_attention_blk(device = xm.xla_device()):
    torch.manual_seed(0)
    B, H, seqQ = 2, 3, 4
    seqPrior, seqActive = 5, 6
    d_head = 64

    # random Q
    Q = torch.randn((B, H, seqQ, d_head), dtype=torch.float16)
    # random prior K,V
    K_prior = torch.randn((B, H, seqPrior, d_head), dtype=torch.float16)
    V_prior = torch.randn((B, H, seqPrior, d_head), dtype=torch.float16)
    # random active K,V
    K_active = torch.randn((B, H, seqActive, d_head), dtype=torch.float16)
    V_active = torch.randn((B, H, seqActive, d_head), dtype=torch.float16)

    past_key_value = (K_prior, V_prior)

    # random boolean masks
    attention_mask = torch.ones((B, H, seqQ, seqPrior), dtype=torch.bool)
    # turn off some random positions
    attention_mask[0, 0, 1, 3] = False
    active_mask = torch.ones((B, H, seqQ, seqActive), dtype=torch.bool)
    active_mask[1, 2, 2, 1] = False

    # 1) reference
    ref_output = compute_for_token_gen_ref(Q, K_active, V_active,
                                           past_key_value,
                                           attention_mask,
                                           active_mask)

    # 2) nki version
    nki_output = compute_for_token_gen_nki(Q, K_active, V_active,
                                           past_key_value,
                                           attention_mask,
                                           active_mask)

    diff = (ref_output - nki_output).abs().max()
    print("Reference shape:", ref_output.shape)
    print("NKI shape:", nki_output.shape)
    print("Max diff =", diff.item())

    # confirm close
    assert diff < 1e-2, f"Mismatch too large: {diff.item()}"
    print("Test PASS")

def main():
    # use Trn1 instance
    device = xm.xla_device()
    # # test SPMD matmul
    # nki_output = test_mlp_gating_fully_optimized_matmul(device)

    # test nki activation fucntion
    test_nki_silu_mul(device)

    # test fused self-attention blk
    # test_self_attention_blk(device)

    # nki_output, lsh, rsh = test_matmul_singletile()
    # nki_output, lsh, rsh = test_block_free()
    # check_match(lsh, rsh, nki_output)
    # performance_test()

if __name__ == "__main__":
    main()
  


 