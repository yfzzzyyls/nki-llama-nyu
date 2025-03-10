# self-attn.py
import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from torch_xla.core import xla_model as xm

###############################################################################
# 1) Fused Self-Attn Kernel for small head dimension
#    with optional user_mask + causal_mask
###############################################################################
@nki.jit
def fused_self_attn_for_SD_small_head_size(
    q_ref,
    k_ref,
    v_ref,
    use_causal_mask=False,
    mixed_precision=True,
    mask_ref=None
):
    """
    Fused self-attention kernel for d_head <= 128.
    Args:
      q_ref, k_ref, v_ref: each shape (seqlen, d_head)
      use_causal_mask: bool => if True, do lower-triangular mask
      mixed_precision: bool => if True, do BF16 + FP32 accumulate
      mask_ref: optional bool tile of shape (seqlen, seqlen). 
                If present, positions where mask_ref[i,j]==False => -9984.0
    """

    kernel_dtype = q_ref.dtype
    pe_in_dt = nl.bfloat16 if mixed_precision else np.float32

    # Basic checks
    seqlen, d_head = q_ref.shape
    assert d_head <= 128, "Cannot use this kernel for d_head>128"
    assert q_ref.shape == (seqlen, d_head)
    assert k_ref.shape == (seqlen, d_head)
    assert v_ref.shape == (seqlen, d_head)
    if mask_ref is not None:
        # Must match [seqlen,seqlen]
        assert mask_ref.shape == (seqlen, seqlen), "mask_ref shape mismatch!"

    # Allocate final output
    out_ref = nl.ndarray((seqlen, d_head), dtype=kernel_dtype, buffer=nl.shared_hbm)

    # Hard-coded scale
    softmax_scale = 0.125

    # Tiling
    q_seq_n_tiles, q_seq_tile_size = seqlen // 128, 128
    k_seq_n_tiles, k_seq_tile_size = seqlen // 128, 128
    d_head_tile_size = d_head
    v_seq_n_tiles, v_seq_tile_size = seqlen // 128, 128

    #----------------------------------
    # 1) Transpose V
    #----------------------------------
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

    #----------------------------------
    # 2) Load + scale Q
    #----------------------------------
    q_local = nl.ndarray(
        (q_seq_n_tiles, nl.par_dim(d_head_tile_size), q_seq_tile_size),
        dtype=pe_in_dt
    )
    ip_q = nl.arange(d_head_tile_size)[:, None]
    if_q = nl.arange(q_seq_tile_size)[None, :]
    for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
        q_local[i_q_seq_tile, ip_q, if_q] = (
            nl.load_transpose2d(
                q_ref[
                    i_q_seq_tile * q_seq_tile_size
                    + nl.arange(q_seq_tile_size)[:, None],
                    nl.arange(d_head_tile_size)[None, :]
                ],
                dtype=pe_in_dt
            ) * softmax_scale
        )

    #----------------------------------
    # 3) Load K
    #----------------------------------
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

    #----------------------------------
    # 4) QK^T => optional mask => softmax => multiply by V
    #----------------------------------
    for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
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

        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            # partial psum
            qk_psum = nl.zeros(
                (nl.par_dim(q_seq_tile_size), k_seq_tile_size),
                dtype=np.float32,
                buffer=nl.psum
            )
            ip_qk = nl.arange(q_seq_tile_size)[:, None]
            if_qk = nl.arange(k_seq_tile_size)[None, :]

            # Dot product
            qk_psum[ip_qk, if_qk] += nisa.nc_matmul(
                moving=k_local[i_k_seq_tile, ip_k, if_k],
                stationary=q_local[i_q_seq_tile, ip_q, if_q]
            )

            # Combine user mask + causal if needed
            pred_bool = None
            if mask_ref is not None:
                # slice from mask_ref => shape=(q_seq_tile_size, k_seq_tile_size)
                pred_bool = mask_ref[
                    i_q_seq_tile * q_seq_tile_size + ip_qk,
                    i_k_seq_tile * k_seq_tile_size + if_qk
                ]
            if use_causal_mask:
                # shape => (q_seq_tile_size, k_seq_tile_size)
                causal_pred = (
                    i_q_seq_tile * q_seq_tile_size + ip_qk
                    >= i_k_seq_tile * k_seq_tile_size + if_qk
                )
                if pred_bool is None:
                    pred_bool = causal_pred
                else:
                    pred_bool = pred_bool & causal_pred

            # If we have final mask => set masked => -9984.0
            if pred_bool is not None:
                qk_res_buf[ip_qk, i_k_seq_tile*k_seq_tile_size + if_qk] = nisa.affine_select(
                    pred=pred_bool,
                    on_true_tile=qk_psum[ip_qk, if_qk],
                    on_false_value=-9984.0,
                    dtype=kernel_dtype
                )
            else:
                qk_res_buf[ip_qk, i_k_seq_tile*k_seq_tile_size + if_qk] = nl.copy(
                    qk_psum[ip_qk, if_qk],
                    dtype=kernel_dtype
                )

            # partial max
            neg_max_res[ip_max, i_k_seq_tile] = nisa.tensor_reduce(
                np.max,
                data=qk_res_buf[ip_qk, i_k_seq_tile*k_seq_tile_size + if_qk],
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

        # Softmax
        ip_softmax = nl.arange(q_seq_tile_size)[:, None]
        if_softmax = nl.arange(seqlen)[None, :]

        softmax_res = nl.ndarray(
            (nl.par_dim(q_seq_tile_size), seqlen),
            dtype=pe_in_dt
        )
        sum_divisor = nl.ndarray(
            (nl.par_dim(q_seq_tile_size), d_head_tile_size),
            dtype=kernel_dtype
        )

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
        ip_sum_res = nl.arange(q_seq_tile_size)[:, None]
        if_sum_res = nl.arange(d_head_tile_size)[None, :]
        sum_divisor[ip_sum_res, if_sum_res] = nl.copy(
            sum_reciprocal_broadcast,
            dtype=kernel_dtype
        )

        # Multiply by V
        trans_softmax_res = nl.ndarray(
            (nl.par_dim(k_seq_tile_size), k_seq_n_tiles, q_seq_tile_size),
            dtype=pe_in_dt
        )
        attn_res_psum = nl.zeros(
            (nl.par_dim(d_head_tile_size), q_seq_tile_size),
            dtype=np.float32,
            buffer=nl.psum
        )

        ip_scores_t = nl.arange(k_seq_tile_size)[:, None]
        if_scores_t = nl.arange(q_seq_tile_size)[None, :]
        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            ip_scores = nl.arange(q_seq_tile_size)[:, None]
            if_scores = nl.arange(k_seq_tile_size)[None, :]
            trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t] = nisa.nc_transpose(
                softmax_res[ip_scores, i_k_seq_tile * k_seq_tile_size + if_scores]
            )

        ip_out = nl.arange(d_head_tile_size)[:, None]
        if_out = nl.arange(q_seq_tile_size)[None, :]
        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
            ip_v_t = nl.arange(k_seq_tile_size)[:, None]
            if_v_t = nl.arange(d_head_tile_size)[None, :]
            attn_res_psum[ip_out, if_out] += nisa.nc_matmul(
                moving=trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t],
                stationary=trans_v[ip_v_t, i_k_seq_tile, if_v_t]
            )

        attn_res_sbuf = nl.copy(attn_res_psum[ip_out, if_out], dtype=kernel_dtype)
        attn_res_div = attn_res_sbuf * nisa.nc_transpose(
            sum_divisor[ip_sum_res, if_sum_res]
        )

        nl.store(
            out_ref[i_q_seq_tile * q_seq_tile_size + if_out, ip_out],
            value=attn_res_div
        )

    return out_ref


###############################################################################
# 2) Replacement for llama.py "compute_for_token_gen" with same signature
###############################################################################
def compute_for_token_gen(
    self,
    Q: torch.Tensor,          # [B, heads, seq_q, d_head]
    K: torch.Tensor,          # [B, heads, seq_active, d_head]
    V: torch.Tensor,          # [B, heads, seq_active, d_head]
    position_ids: torch.Tensor,
    past_key_value: Tuple[torch.Tensor, torch.Tensor],  # (K_prev, V_prev)
    attention_mask: Optional[torch.Tensor],
    active_mask: torch.Tensor
) -> torch.Tensor:
    """
    Drop-in replacement for llama.py's compute_for_token_gen(...).
    - Merge old + new K/V
    - Build 2D user mask => (seq_q, seq_k)
    - Pad Q,K,V,mask to multiples of 128
    - Call fused_self_attn_for_SD_small_head_size(..., mask_ref=..., use_causal_mask=True)
    - Slice back to real seq_q
    - Return final shape [B, heads, seq_q, d_head]
    """
    # 1) Merge old + new:
    K_prev, V_prev = past_key_value
    # If you have Grouped Query Attention, replicate accordingly:
    # K_prev = repeat_kv(K_prev, self.num_key_value_groups)
    # V_prev = repeat_kv(V_prev, self.num_key_value_groups)
    # K = repeat_kv(K, self.num_key_value_groups)
    # V = repeat_kv(V, self.num_key_value_groups)
    # For demonstration, we skip the GQA repetition if not needed.

    K_cat = torch.cat([K_prev, K], dim=2)  # [B, heads, seq_prev+seq_active, d_head]
    V_cat = torch.cat([V_prev, V], dim=2)

    B, H, seq_q, d_head = Q.shape
    seq_k = K_cat.shape[2]

    attn_out = torch.empty_like(Q)  # [B, heads, seq_q, d_head]

    # 2) Pad up to multiples of 128 for the fused kernel
    padded_len = max(seq_q, seq_k, 128)
    if padded_len % 128 != 0:
        padded_len = ((padded_len + 127) // 128) * 128

    # 3) For each (batch, head), slice out Q,K,V => 2D, build/pad mask => call fused kernel
    for b in range(B):
        for h in range(H):
            q_2d = Q[b, h]        # (seq_q, d_head)
            k_2d = K_cat[b, h]    # (seq_k, d_head)
            v_2d = V_cat[b, h]    # (seq_k, d_head)

            # Build a 2D mask => shape (seq_q, seq_k) if present
            if attention_mask is not None:
                user_mask_2d = attention_mask[b, h]  # shape (seq_q, seq_k)
            else:
                user_mask_2d = None

            # -- Pad Q
            if seq_q < padded_len:
                q_2d_pad = F.pad(q_2d, (0, 0, 0, padded_len - seq_q))
            else:
                q_2d_pad = q_2d

            # -- Pad K,V
            if seq_k < padded_len:
                k_2d_pad = F.pad(k_2d, (0, 0, 0, padded_len - seq_k))
                v_2d_pad = F.pad(v_2d, (0, 0, 0, padded_len - seq_k))
            else:
                k_2d_pad = k_2d
                v_2d_pad = v_2d

            # -- Pad the 2D mask => (padded_len, padded_len)
            if user_mask_2d is not None:
                user_mask_2d_pad = F.pad(
                    user_mask_2d,
                    (0, padded_len - seq_k, 0, padded_len - seq_q),
                    value=False
                )
            else:
                user_mask_2d_pad = None

            # 4) Call the fused kernel => shape (padded_len, d_head)
            # We'll do an autoregressive decode => use_causal_mask=True
            out_2d_pad = fused_self_attn_for_SD_small_head_size(
                q_ref=q_2d_pad,
                k_ref=k_2d_pad,
                v_ref=v_2d_pad,
                use_causal_mask=True,
                mixed_precision=True,
                mask_ref=user_mask_2d_pad
            )
            # Slice back the top seq_q rows
            attn_out[b, h] = out_2d_pad[:seq_q]

    return attn_out


###############################################################################
# 3) Test function (optional) to confirm correctness
###############################################################################
def test_compute_for_token_gen():
    # XLA device
    device = xm.xla_device()
    torch.manual_seed(0)

    B, H = 2, 3
    seq_prev = 4
    seq_active = 2
    seq_q = 2
    d_head = 64  # <=128

    Q = torch.randn(B, H, seq_q, d_head, device=device, dtype=torch.float32)
    K_prev = torch.randn(B, H, seq_prev, d_head, device=device, dtype=torch.float32)
    V_prev = torch.randn(B, H, seq_prev, d_head, device=device, dtype=torch.float32)
    K_new = torch.randn(B, H, seq_active, d_head, device=device, dtype=torch.float32)
    V_new = torch.randn(B, H, seq_active, d_head, device=device, dtype=torch.float32)

    # Build a 2D attention_mask => shape (B,H,seq_q,seq_prev+seq_active)
    seq_k = seq_prev + seq_active
    attn_mask = (torch.rand(B, H, seq_q, seq_k, device=device) > 0.5)

    # Dummy placeholders for signature
    position_ids = torch.zeros((B, seq_q), device=device, dtype=torch.long)
    past_key_value = (K_prev, V_prev)
    active_mask = torch.empty(0, device=device)

    # Call compute_for_token_gen
    out_nki = compute_for_token_gen(
        self=None,    # if we don't actually need "self", pass None or a dummy
        Q=Q,
        K=K_new,
        V=V_new,
        position_ids=position_ids,
        past_key_value=past_key_value,
        attention_mask=attn_mask,
        active_mask=active_mask
    )
    print("[TEST] out_nki shape:", out_nki.shape)


if __name__ == "__main__":
    test_compute_for_token_gen()