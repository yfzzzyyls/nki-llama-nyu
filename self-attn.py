import copy
import sys
import gc
import logging
import math
import numpy as np
import time
from typing import List, Optional, Tuple, Type

import torch
import torch.nn.functional as F
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

@nki.jit
def fused_self_attn_for_SD_small_head_size(
    q_ref,
    k_ref,
    v_ref,
    use_causal_mask=False,
    mixed_precision=True
):
    """
    Fused self attention kernel for small head dimension Stable Diffusion workload,
    simplified for this tutorial.

    Computes softmax(QK^T)V. Decoder model can optionally include a causal mask
    application. Does not include QKV projection, output projection, dropout,
    residual connection, etc.

    This kernel is designed to be used for Stable Diffusion models where the
    d_head is smaller or equal to 128. Assertion is thrown if `d_head` does
    not satisfy the requirement.

    IO tensor layouts:
      - q_ref: shape (seq_q, d_head)
      - k_ref: shape (seq_k, d_head)
      - v_ref: shape (seq_v, d_head)
      - out_ref: shape (seq_q, d_head)
      - We use seq_q and seq_k and seq_v just for clarity, this kernel
        requires seq_q == seq_k == seq_v.

    IO tensor dtypes:
      - This kernel assumes all IO tensors have the same dtype.
      - If mixed_precision is True, then all Tensor Engine operations
        will be performed in bfloat16 and accumulation will be performed
        in float32. Otherwise, the intermediates will be in the same
        type as the inputs.
    """
    # Use q_ref dtype as the intermediate tensor dtype
    # Assume all IO tensors have the same dtype
    kernel_dtype = q_ref.dtype
    pe_in_dt = nl.bfloat16 if mixed_precision else np.float32
    assert q_ref.dtype == k_ref.dtype == v_ref.dtype

    # Shape checking
    seqlen, d_head = q_ref.shape
    assert d_head <= 128, "Cannot use this kernel for d_head > 128"
    assert tuple(q_ref.shape) == (seqlen, d_head), "Input shape mismatch!"
    assert tuple(k_ref.shape) == (seqlen, d_head), "Input shape mismatch!"
    assert tuple(v_ref.shape) == (seqlen, d_head), (
        f"Input shape mismatch! Expected: {(seqlen, d_head)} "
        f"Actual: {tuple(v_ref.shape)}"
    )
    out_ref = nl.ndarray((seqlen, d_head), dtype=q_ref.dtype, buffer=nl.shared_hbm)

    # Softmax scaling factor, multiplied onto Q
    softmax_scale = 0.125

    q_seq_n_tiles, q_seq_tile_size = seqlen // 128, 128
    k_seq_n_tiles, k_seq_tile_size = seqlen // 128, 128
    # No tiling on d_head dimension since the dimension of d_head fits in SB
    d_head_tile_size = d_head
    v_seq_n_tiles, v_seq_tile_size = seqlen // 128, 128

    # -----------------------------------------------
    # Step 1. transpose(tensor_v)
    # -----------------------------------------------
    # Buffer for v matrix transposed
    # Pre-fetch and keep it in SBUF throughout different softmax tiles
    trans_v = nl.ndarray(
        (nl.par_dim(v_seq_tile_size), v_seq_n_tiles, d_head),
        dtype=pe_in_dt
    )

    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
        ip_v = nl.arange(v_seq_tile_size)[:, None]
        if_v = nl.arange(d_head_tile_size)[None, :]
        trans_v[ip_v, i_k_seq_tile, if_v] = nl.load(
            v_ref[i_k_seq_tile * k_seq_tile_size + ip_v, if_v],
            dtype=pe_in_dt
        )

    # -----------------------------------------------
    # Prepare local Q tiles
    # -----------------------------------------------
    q_local = nl.ndarray(
        (q_seq_n_tiles, nl.par_dim(d_head_tile_size), q_seq_tile_size),
        dtype=pe_in_dt
    )
    ip_q = nl.arange(d_head_tile_size)[:, None]
    if_q = nl.arange(q_seq_tile_size)[None, :]
    for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
        q_local[i_q_seq_tile, ip_q, if_q] = nl.load_transpose2d(
            q_ref[
                i_q_seq_tile * q_seq_tile_size
                + nl.arange(q_seq_tile_size)[:, None],
                nl.arange(d_head_tile_size)[None, :]
            ],
            dtype=pe_in_dt
        ) * softmax_scale

    # -----------------------------------------------
    # Prepare local K tiles
    # -----------------------------------------------
    k_local = nl.ndarray(
        (k_seq_n_tiles, nl.par_dim(d_head_tile_size), k_seq_tile_size),
        dtype=pe_in_dt
    )
    ip_k = nl.arange(d_head_tile_size)[:, None]
    if_k = nl.arange(k_seq_tile_size)[None, :]
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
        k_local[i_k_seq_tile, ip_k, if_k] = nl.load_transpose2d(
            k_ref[
                i_k_seq_tile * k_seq_tile_size
                + nl.arange(k_seq_tile_size)[:, None],
                nl.arange(d_head_tile_size)[None, :]
            ],
            dtype=pe_in_dt
        )

    # -----------------------------------------------
    # Perform QK^T + optional mask + softmax + multiply by V
    # -----------------------------------------------
    for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
        # A SBUF buffer for an independent softmax tile
        qk_res_buf = nl.ndarray(
            (nl.par_dim(q_seq_tile_size), seqlen),
            dtype=kernel_dtype
        )
        neg_max_res = nl.ndarray(
            (nl.par_dim(q_seq_tile_size), k_seq_n_tiles),
            dtype=kernel_dtype
        )
        ip_max = nl.arange(q_seq_tile_size)[:, None]
        if_max = nl.arange(k_seq_n_tiles)[None, :]

        # Loop over RHS free of matmul(stationary=tensor_q, moving=tensor_k, contract=d_head)
        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            # PSUM buffer shape: [q_seq_tile_size P, k_seq_tile_size F]
            qk_psum = nl.zeros(
                (nl.par_dim(q_seq_tile_size), k_seq_tile_size),
                dtype=np.float32,
                buffer=nl.psum
            )
            ip_qk = nl.arange(q_seq_tile_size)[:, None]
            if_qk = nl.arange(k_seq_tile_size)[None, :]

            # Step 2. matmul(stationary=tensor_q, moving=tensor_k, contract=d_head)
            qk_psum[ip_qk, if_qk] += nisa.nc_matmul(
                moving=k_local[i_k_seq_tile, ip_k, if_k],
                stationary=q_local[i_q_seq_tile, ip_q, if_q]
            )

            # Step 3. Apply optional causal mask
            if use_causal_mask:
                # Magic number -9984.0 to replace -inf (like neuronx-cc)
                qk_res_buf[
                    ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk
                ] = nisa.affine_select(
                    pred=(
                        i_q_seq_tile * q_seq_tile_size + ip_qk
                        >= i_k_seq_tile * k_seq_tile_size + if_qk
                    ),
                    on_true_tile=qk_psum[ip_qk, if_qk],
                    on_false_value=-9984.0,
                    dtype=kernel_dtype
                )
            else:
                # Simply send psum result back to sbuf
                qk_res_buf[
                    ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk
                ] = nl.copy(qk_psum[ip_qk, if_qk], dtype=kernel_dtype)

            # Step 4. Softmax (partial, track negative max)
            neg_max_res[ip_max, i_k_seq_tile] = nisa.tensor_reduce(
                np.max,
                data=qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk],
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

        ip_softmax = nl.arange(q_seq_tile_size)[:, None]
        if_softmax = nl.arange(seqlen)[None, :]
        ip_sum_res = nl.arange(q_seq_tile_size)[:, None]
        if_sum_res = nl.arange(d_head_tile_size)[None, :]

        softmax_res = nl.ndarray(
            (nl.par_dim(q_seq_tile_size), seqlen),
            dtype=pe_in_dt
        )
        sum_divisor = nl.ndarray(
            (nl.par_dim(q_seq_tile_size), d_head_tile_size),
            dtype=kernel_dtype
        )

        # Exponential + reduce
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
        softmax_res[ip_softmax, if_softmax] = nl.copy(exp_res, dtype=pe_in_dt)

        sum_reciprocal_broadcast = (1.0 / sum_res).broadcast_to(
            (q_seq_tile_size, d_head_tile_size)
        )
        sum_divisor[ip_sum_res, if_sum_res] = nl.copy(
            sum_reciprocal_broadcast,
            dtype=kernel_dtype
        )

        # Buffer for transposed softmax results (FP32 in PSUM)
        trans_softmax_res = nl.ndarray(
            (nl.par_dim(k_seq_tile_size), k_seq_n_tiles, q_seq_tile_size),
            dtype=pe_in_dt
        )

        # Result psum buffer has the hidden dim as P
        attn_res_psum = nl.zeros(
            (nl.par_dim(d_head_tile_size), q_seq_tile_size),
            dtype=np.float32,
            buffer=nl.psum
        )

        ip_scores_t = nl.arange(k_seq_tile_size)[:, None]
        if_scores_t = nl.arange(q_seq_tile_size)[None, :]

        # Step 5. transpose(softmax_res) for each tile
        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            ip_scores = nl.arange(q_seq_tile_size)[:, None]
            if_scores = nl.arange(k_seq_tile_size)[None, :]
            trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t] = nisa.nc_transpose(
                softmax_res[ip_scores, i_k_seq_tile * k_seq_tile_size + if_scores]
            )

        # Step 6. matmul_1(stationary=trans_v, moving=trans_softmax_res)
        ip_out = nl.arange(d_head_tile_size)[:, None]
        if_out = nl.arange(q_seq_tile_size)[None, :]
        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            ip_v_t = nl.arange(k_seq_tile_size)[:, None]
            if_v_t = nl.arange(d_head_tile_size)[None, :]
            attn_res_psum[ip_out, if_out] += nisa.nc_matmul(
                moving=trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t],
                stationary=trans_v[ip_v_t, i_k_seq_tile, if_v_t]
            )

        attn_res_sbuf = nl.copy(
            attn_res_psum[ip_out, if_out],
            dtype=kernel_dtype
        )
        attn_res_div = attn_res_sbuf * nisa.nc_transpose(
            sum_divisor[ip_sum_res, if_sum_res]
        )

        nl.store(
            out_ref[i_q_seq_tile * q_seq_tile_size + if_out, ip_out],
            value=attn_res_div
        )

    return out_ref


################################################################################
# 2) REFERENCE IMPLEMENTATION FOR TOKEN GEN (Incremental) ATTENTION
#    This mimics "token gen" logic: we have Q, and we want to attend over 
#    past K/V plus new K/V. We'll do a simple PyTorch approach for correctness.
################################################################################
def compute_for_token_gen_ref(
    Q,           # (batch, heads, seq_q, d_head)
    K_prev,      # (batch, heads, seq_prev, d_head)
    K_active,    # (batch, heads, seq_active, d_head)
    V_prev,      # (batch, heads, seq_prev, d_head)
    V_active,    # (batch, heads, seq_active, d_head)
    attention_mask=None,  # bool mask with shape (batch, heads, seq_q, seq_q+seq_prev)
    use_causal_mask=False,
):
    """
    A reference PyTorch impl for incremental decode:
      - Combine old K/V with new K/V
      - Q * K^T
      - Apply mask
      - Softmax
      - Multiply by V
    We'll just do a naive approach in float32 for correctness.
    """
    # Combine old and new K
    K_cat = torch.cat([K_prev, K_active], dim=2)   # shape (B,H, seq_prev+seq_active, d_head)
    V_cat = torch.cat([V_prev, V_active], dim=2)

    B, H, seq_q, d_head = Q.shape
    seq_k = K_cat.shape[2]

    # We'll do: attn_scores = Q @ K^T / sqrt(d_head)
    # Then mask, then softmax, then multiply by V
    Q_2d = Q.view(B*H, seq_q, d_head)
    K_2d = K_cat.view(B*H, seq_k, d_head)
    # shape = (B*H, seq_q, seq_k)
    attn_scores = torch.bmm(Q_2d, K_2d.transpose(1,2)) / math.sqrt(d_head)

    if attention_mask is not None:
        # Where mask==False => set to large negative
        attn_scores = torch.where(
            attention_mask.view(B*H, seq_q, seq_k),
            attn_scores,
            torch.tensor(-1e9, dtype=attn_scores.dtype, device=attn_scores.device)
        )
    elif use_causal_mask:
        # We'll do a simple lower-triangular (including diagonal) mask
        # We assume 'seq_q==seq_k' if purely autoregressive.
        # For clarity, we build a [seq_q, seq_k] mask with True=valid, 
        # then broadcast across B*H
        mask_2d = torch.ones(seq_q, seq_k, dtype=torch.bool, device=Q.device)
        for iq in range(seq_q):
            for ik in range(seq_k):
                if iq < ik:
                    mask_2d[iq, ik] = False
        attn_scores = torch.where(
            mask_2d.unsqueeze(0),
            attn_scores,
            torch.tensor(-1e9, dtype=attn_scores.dtype, device=attn_scores.device)
        )

    attn_weights = torch.softmax(attn_scores.float(), dim=-1).to(Q.dtype)

    V_2d = V_cat.view(B*H, seq_k, d_head)
    attn_output_2d = torch.bmm(attn_weights, V_2d)  # shape (B*H, seq_q, d_head)
    attn_output = attn_output_2d.view(B, H, seq_q, d_head)
    return attn_output


################################################################################
# 3) NKI-BASED IMPLEMENTATION OF "compute_for_token_gen"
#    We'll unify old & new K/V, flatten each batch/head, call the fused kernel.
################################################################################

import torch
import torch.nn.functional as F
import math

def compute_for_token_gen_nki(
    Q,           # (batch, heads, seq_q, d_head)
    K_prev,      # (batch, heads, seq_prev, d_head)
    K_active,    # (batch, heads, seq_active, d_head)
    V_prev,      # (batch, heads, seq_prev, d_head)
    V_active,    # (batch, heads, seq_active, d_head)
    attention_mask=None,  # bool or None
    use_causal_mask=False
):
    """
    Use the fused_self_attn_for_SD_small_head_size kernel, 
    forcing the sequence dimension up to a multiple of 128 so 
    that the kernel's loops run.
    """
    device = Q.device
    dtype = Q.dtype
    B, H, seq_q, d_head = Q.shape

    # Concatenate old & new keys/values:
    K_cat = torch.cat([K_prev, K_active], dim=2)  # shape = (B,H, seq_k, d_head)
    V_cat = torch.cat([V_prev, V_active], dim=2)
    seq_k = K_cat.shape[2]

    # We'll produce [B, H, seq_q, d_head] as our final output
    attn_out = torch.empty_like(Q)

    # 1) Determine the final padded_len so seq_q and seq_k 
    #    become the same multiple-of-128 dimension:
    padded_len = max(seq_q, seq_k)

    # If you want a minimum of 128:
    if padded_len < 128:
        padded_len = 128

    # Now round up to the nearest multiple of 128:
    if padded_len % 128 != 0:
        padded_len = ((padded_len + 127) // 128) * 128

    for b in range(B):
        for h in range(H):
            # Extract Q, K, V slices for this (b,h)
            q_2d = Q[b, h]            # shape (seq_q, d_head)
            k_2d = K_cat[b, h]        # shape (seq_k, d_head)
            v_2d = V_cat[b, h]        # shape (seq_k, d_head)

            # 2) Pad Q, K, V so each becomes (padded_len, d_head)
            q_2d_padded = F.pad(
                q_2d, (0, 0, 0, padded_len - seq_q)
            ) if seq_q < padded_len else q_2d

            k_2d_padded = F.pad(
                k_2d, (0, 0, 0, padded_len - seq_k)
            ) if seq_k < padded_len else k_2d

            v_2d_padded = F.pad(
                v_2d, (0, 0, 0, padded_len - seq_k)
            ) if seq_k < padded_len else v_2d

            # 3) Call the fused kernel with the padded shapes
            out_2d_padded = fused_self_attn_for_SD_small_head_size(
                q_2d_padded,
                k_2d_padded,
                v_2d_padded,
                use_causal_mask=use_causal_mask,
                mixed_precision=True,
            )
            # out_2d_padded is shape (padded_len, d_head)

            # 4) Slice out only the first seq_q rows => final shape (seq_q, d_head)
            out_2d = out_2d_padded[:seq_q]
            attn_out[b, h] = out_2d.to(dtype=dtype)

    return attn_out

################################################################################
# 4) TEST FUNCTION:
#    We'll just do a small example and verify correctness w.r.t. 
#    compute_for_token_gen_ref
################################################################################

def test_compute_for_token_gen():
    # 1) Pick the XLA device
    device = xm.xla_device()  # or torch.device("privateuseone")

    torch.manual_seed(0)

    B, H = 2, 3
    seq_prev, seq_active, seq_q = 4, 2, 2
    d_head = 64  # must be <=128 for fused kernel

    # 2) Generate data on CPU, then move to XLA device
    Q = torch.randn(B, H, seq_q, d_head, dtype=torch.float32).to(device)
    K_prev = torch.randn(B, H, seq_prev, d_head, dtype=torch.float32).to(device)
    K_active = torch.randn(B, H, seq_active, d_head, dtype=torch.float32).to(device)
    V_prev = torch.randn(B, H, seq_prev, d_head, dtype=torch.float32).to(device)
    V_active = torch.randn(B, H, seq_active, d_head, dtype=torch.float32).to(device)

    # 3) Run your reference code (it can be CPU or XLA, but for fair comparison, 
    #    you might also do .to(device) to keep everything consistent)
    out_ref = compute_for_token_gen_ref(
        Q, K_prev, K_active, V_prev, V_active,
        attention_mask=None,
        use_causal_mask=True
    )

    # 4) Run NKI-based kernel call (now all on XLA device)
    out_nki = compute_for_token_gen_nki(
        Q, K_prev, K_active, V_prev, V_active,
        attention_mask=None,
        use_causal_mask=True
    )

    # 5) For printing or comparing on CPU, move results back
    out_ref_cpu = out_ref.cpu()
    out_nki_cpu = out_nki.cpu()

    max_abs_diff = (out_ref_cpu - out_nki_cpu).abs().max().item()
    print(f"[TEST] max_abs_diff between reference & NKI kernel = {max_abs_diff:.6e}")
    print("Ref output:", out_ref_cpu[0, 0, 0, :8])
    print("NKI output:", out_nki_cpu[0, 0, 0, :8])


if __name__ == "__main__":
    test_compute_for_token_gen()