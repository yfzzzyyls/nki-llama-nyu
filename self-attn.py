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
    Fused self-attention kernel for d_head <=128.

    Q,K,V => shape (seqlen, d_head)
    mask_ref => shape (seqlen,seqlen), bool (or None).
    If an element is False => that QK entry => -9984.
    If use_causal_mask=True => also i>=j => else => -9984.

    This version:
      - Uses plain Python "range(...)" for loops over tile blocks, so i_q_seq_tile
        and i_k_seq_tile become normal Python ints.
      - Inside each block, we allocate local SBUF tiles and do nested loops in Python
        over range(128) for row & col to fill single elements.

    The big difference: we do *no* nki symbolic loops for i_q_seq_tile or rr, so
    there's no risk of "got 'mask' instead" from row_abs >= col_abs.
    """

    kernel_dtype = q_ref.dtype
    pe_in_dt = nl.bfloat16 if mixed_precision else np.float32

    seqlen, d_head = q_ref.shape
    assert d_head <= 128
    assert (k_ref.shape == (seqlen, d_head))
    assert (v_ref.shape == (seqlen, d_head))
    if mask_ref is not None:
        assert (mask_ref.shape == (seqlen, seqlen))

    # Output
    out_ref = nl.ndarray((seqlen, d_head), dtype=kernel_dtype, buffer=nl.shared_hbm)

    softmax_scale = 0.125
    # how many tile blocks of size 128
    q_seq_n_tiles = seqlen // 128
    k_seq_n_tiles = seqlen // 128
    tile_size = 128

    # (1) SBUF tiles for "True" & "False" => shape=(128,128)
    ones_bool_tile  = nl.ndarray((tile_size, tile_size), dtype=bool, buffer=nl.sbuf)
    zeros_bool_tile = nl.ndarray((tile_size, tile_size), dtype=bool, buffer=nl.sbuf)
    for rr in range(tile_size):
        for cc in range(tile_size):
            ones_bool_tile[rr, cc] = True
            zeros_bool_tile[rr, cc] = False

    # (2) SBUF tile for "neg_inf" => shape=(128,128)
    neg_inf_tile = nl.ndarray((tile_size, tile_size), dtype=kernel_dtype, buffer=nl.sbuf)
    for rr in range(tile_size):
        for cc in range(tile_size):
            neg_inf_tile[rr, cc] = -9984.0

    # (3) Transpose V blocks
    trans_v = nl.ndarray(
        (nl.par_dim(tile_size), k_seq_n_tiles, d_head),
        dtype=pe_in_dt
    )
    for i_k_seq_tile in range(k_seq_n_tiles):  # plain python
        # load chunk
        for rr in range(tile_size):
            for cc in range(d_head):
                row_abs = i_k_seq_tile*tile_size + rr
                val = nl.load(k_ref[row_abs, cc], dtype=k_ref.dtype)
                trans_v[rr, i_k_seq_tile, cc] = val

    # (4) Q => local blocks
    q_local = nl.ndarray(
        (q_seq_n_tiles, nl.par_dim(d_head), tile_size),
        dtype=pe_in_dt
    )
    for i_q_seq_tile in range(q_seq_n_tiles):
        # we do a nested loop to load 128x(d_head) from q_ref => transpose => scale
        for rr in range(d_head):
            for cc in range(tile_size):
                row_abs = i_q_seq_tile*tile_size + cc
                val = nl.load(q_ref[row_abs, rr], dtype=q_ref.dtype)
                # scale
                scaled = val * softmax_scale
                # store transposed => q_local[i_q_seq_tile, rr, cc]
                q_local[i_q_seq_tile, rr, cc] = scaled

    # (5) K => local blocks
    k_local = nl.ndarray(
        (k_seq_n_tiles, nl.par_dim(d_head), tile_size),
        dtype=pe_in_dt
    )
    for i_k_seq_tile in range(k_seq_n_tiles):
        for rr in range(d_head):
            for cc in range(tile_size):
                row_abs = i_k_seq_tile*tile_size + cc
                val = nl.load(k_ref[row_abs, rr], dtype=k_ref.dtype)
                k_local[i_k_seq_tile, rr, cc] = val

    # (6) Now do QK => optional mask => -9984 => softmax => * V
    for i_q_seq_tile in range(q_seq_n_tiles):
        # qk_res_buf => shape=(128, seqlen)
        qk_res_buf = nl.ndarray((nl.par_dim(tile_size), seqlen), dtype=kernel_dtype)
        neg_max_res = nl.ndarray((nl.par_dim(tile_size), k_seq_n_tiles), dtype=kernel_dtype)

        for i_k_seq_tile in range(k_seq_n_tiles):
            # Dot product => shape=(128,128)
            qk_psum = nl.zeros((nl.par_dim(tile_size), tile_size), dtype=np.float32, buffer=nl.psum)

            # do matmul => qk_psum
            # we can do an nisa.nc_matmul => we pass tile references
            # first, let's do a tile-level matmul: q_local[i_q_seq_tile,...], k_local[i_k_seq_tile,...]
            qk_psum += nisa.nc_matmul(
                moving=k_local[i_k_seq_tile],  # shape (d_head, 128)
                stationary=q_local[i_q_seq_tile]  # shape (d_head, 128)
            )

            # (A) Combine user mask + causal => build final bool tile in SBUF
            final_mask_sbuf = nl.ndarray((tile_size, tile_size), dtype=bool, buffer=nl.sbuf)
            # initialize to True
            for rr in range(tile_size):
                for cc in range(tile_size):
                    final_mask_sbuf[rr, cc] = True

            # user mask
            if mask_ref is not None:
                user_sbuf = nl.ndarray((tile_size, tile_size), dtype=bool, buffer=nl.sbuf)
                # fill from mask_ref
                for rr in range(tile_size):
                    for cc in range(tile_size):
                        row_abs = i_q_seq_tile*tile_size + rr
                        col_abs = i_k_seq_tile*tile_size + cc
                        val = nl.load(mask_ref[row_abs, col_abs], dtype=bool)
                        user_sbuf[rr, cc] = val
                # AND
                for rr in range(tile_size):
                    for cc in range(tile_size):
                        final_mask_sbuf[rr, cc] = final_mask_sbuf[rr, cc] & user_sbuf[rr, cc]

            # causal
            if use_causal_mask:
                causal_sbuf = nl.ndarray((tile_size, tile_size), dtype=bool, buffer=nl.sbuf)
                for rr in range(tile_size):
                    for cc in range(tile_size):
                        row_abs = i_q_seq_tile*tile_size + rr
                        col_abs = i_k_seq_tile*tile_size + cc
                        # pure python int => pure python bool
                        causal_val = (row_abs >= col_abs)
                        causal_sbuf[rr, cc] = causal_val
                # AND
                for rr in range(tile_size):
                    for cc in range(tile_size):
                        final_mask_sbuf[rr, cc] = final_mask_sbuf[rr, cc] & causal_sbuf[rr, cc]

            # (B) affine_select => shape=(128,128)
            masked_tile = nisa.affine_select(
                pred=final_mask_sbuf,       # shape=(128,128)
                on_true_tile=qk_psum,       # shape=(128,128)
                on_false_tile=neg_inf_tile, # shape=(128,128)
                dtype=kernel_dtype
            )

            # now store masked_tile => qk_res_buf => columns => i_k_seq_tile block
            # a nested loop again
            for rr in range(tile_size):
                for cc in range(tile_size):
                    # row in the partial => rr
                    # col in the partial => i_k_seq_tile*128 + cc
                    col_abs = i_k_seq_tile*tile_size + cc
                    qk_res_buf[rr, col_abs] = masked_tile[rr, cc]

            # partial rowwise max => we'll do a rowwise approach using nisa.tensor_reduce
            # we can do it tile by tile
            # allocate a small tile => shape=(128)
            partial_max = nl.ndarray((nl.par_dim(tile_size),), dtype=kernel_dtype)
            partial_max += nisa.tensor_reduce(
                np.max,
                data=masked_tile,
                axis=(1,),
                dtype=kernel_dtype,
                negate=True
            )
            neg_max_res[range(tile_size), i_k_seq_tile] = partial_max[range(tile_size)]

        # min => across i_k_seq_tile dimension
        # shape => (128, k_seq_n_tiles)
        final_min = nisa.tensor_reduce(
            np.min,
            data=neg_max_res[range(tile_size), range(k_seq_n_tiles)],
            axis=(1,),
            dtype=kernel_dtype,
            negate=False
        )

        # (C) Softmax
        # shape => (128, seqlen)
        # we do exp => row by row => then row sum
        # let's do nisa.activation => we can do a big tile approach
        # for simplicity, let's do it in one chunk
        exp_res = nisa.activation(
            np.exp,
            data=qk_res_buf[range(tile_size), range(seqlen)],
            bias=final_min,
            scale=1.0
        )
        # sum => rowwise
        sum_res = nisa.tensor_reduce(
            np.add,
            data=exp_res,
            axis=(1,),
            dtype=kernel_dtype
        )
        # We'll store the final => shape => (128,seqlen)
        softmax_res = nl.ndarray((nl.par_dim(tile_size), seqlen), dtype=pe_in_dt)
        softmax_res[range(tile_size), range(seqlen)] = nl.copy(exp_res, dtype=pe_in_dt)

        # 1/sum => shape => (128)
        # we broadcast => (128,d_head)
        sum_divisor = nl.ndarray((nl.par_dim(tile_size), d_head), dtype=kernel_dtype)
        # replicate
        for r in range(tile_size):
            # 1.0 / sum_res[r]
            inv_val = 1.0 / sum_res[r]
            for c in range(d_head):
                sum_divisor[r, c] = inv_val

        # (D) Multiply by V
        # We'll do a tile-level transpose => shape => (128, seqlen) => (k_seq_n_tiles blocks)
        trans_softmax_res = nl.ndarray(
            (nl.par_dim(tile_size), k_seq_n_tiles, tile_size),
            dtype=pe_in_dt
        )
        for i_k_seq_tile in range(k_seq_n_tiles):
            # columns => i_k_seq_tile block => range(i_k_seq_tile*128, i_k_seq_tile*128+128)
            # we'll do an nisa.nc_transpose => input => (128,128)
            # We'll do a nested copy approach for brevity
            block_sbuf = nl.ndarray((nl.par_dim(tile_size), tile_size), dtype=pe_in_dt, buffer=nl.sbuf)
            for rr in range(tile_size):
                for cc in range(tile_size):
                    col_abs = i_k_seq_tile*tile_size + cc
                    block_sbuf[rr, cc] = softmax_res[rr, col_abs]

            trans_block = nisa.nc_transpose(block_sbuf)
            trans_softmax_res[range(tile_size), i_k_seq_tile, range(tile_size)] = trans_block[range(tile_size), range(tile_size)]

        # matmul with trans_v => shape => (d_head,128)
        attn_res_psum = nl.zeros((nl.par_dim(d_head), tile_size), dtype=np.float32, buffer=nl.psum)
        for i_k_seq_tile in range(k_seq_n_tiles):
            attn_res_psum += nisa.nc_matmul(
                moving=trans_softmax_res[range(tile_size), i_k_seq_tile, range(tile_size)],
                stationary=trans_v[range(tile_size), i_k_seq_tile, range(d_head)]
            )

        # multiply => sum_divisor => shape => (128,d_head) => need to do transpose => we do a tile approach
        # final => shape => (128,d_head)
        attn_res_sbuf = nl.copy(attn_res_psum[range(d_head), range(tile_size)], dtype=kernel_dtype)
        # do an elementwise multiply with the transpose of sum_divisor => we can do a nested loop
        # for brevity, let's do a tile-level approach => nisa.nc_transpose => then nisa.nc_mul
        sum_div_t = nisa.nc_transpose(sum_divisor[range(tile_size), range(d_head)])
        # shape => (d_head,128)
        # shape attn_res_sbuf => (d_head,128)
        scaled_attn = nl.zeros((nl.par_dim(d_head), tile_size), dtype=kernel_dtype, buffer=nl.sbuf)
        scaled_attn += nisa.nc_mul(
            attn_res_sbuf,
            sum_div_t
        )

        # store => out_ref => rows => i_q_seq_tile block
        for rr in range(tile_size):
            row_abs = i_q_seq_tile*tile_size + rr
            for cc in range(d_head):
                out_ref[row_abs, cc] = scaled_attn[cc, rr]

    return out_ref


def compute_for_token_gen(
    self,
    Q: torch.Tensor,       # [B, heads, seq_q, d_head]
    K: torch.Tensor,       # [B, heads, seq_active, d_head]
    V: torch.Tensor,       # [B, heads, seq_active, d_head]
    position_ids: torch.Tensor,
    past_key_value: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """
    1) Merge old + new K,V => shape (B,H, seq_k, d_head)
    2) If attention_mask => shape(B,H,seq_q,seq_k), bool
    3) pad Q,K,V + mask => multiples of 128
    4) call fused_self_attn_for_SD_small_head_size => shape (padded_len,d_head)
    5) slice => (seq_q,d_head)
    6) return => (B,H,seq_q,d_head)
    """
    B,H,seq_q,d_head = Q.shape
    K_prev, V_prev = past_key_value
    K_cat = torch.cat([K_prev, K], dim=2)  # shape => (B,H,seq_k,d_head)
    V_cat = torch.cat([V_prev, V], dim=2)
    seq_k = K_cat.shape[2]
    attn_out = torch.empty_like(Q)

    if attention_mask is not None:
        attention_mask = attention_mask.bool()

    padded_len = max(seq_q, seq_k, 128)
    if padded_len % 128 != 0:
        padded_len = ((padded_len + 127)//128)*128

    for b in range(B):
        for h_ in range(H):
            q_2d = Q[b,h_]
            k_2d = K_cat[b,h_]
            v_2d = V_cat[b,h_]
            if attention_mask is not None:
                user_mask_2d = attention_mask[b,h_]
            else:
                user_mask_2d = None

            # pad Q => (padded_len,d_head)
            if seq_q < padded_len:
                q_2d_pad = F.pad(q_2d,(0,0,0,padded_len-seq_q))
            else:
                q_2d_pad = q_2d
            # pad K,V => (padded_len,d_head)
            if seq_k < padded_len:
                k_2d_pad = F.pad(k_2d,(0,0,0,padded_len-seq_k))
                v_2d_pad = F.pad(v_2d,(0,0,0,padded_len-seq_k))
            else:
                k_2d_pad = k_2d
                v_2d_pad = v_2d

            if user_mask_2d is not None:
                user_mask_2d_pad = F.pad(
                    user_mask_2d,
                    (0,padded_len-seq_k,0,padded_len-seq_q),
                    value=False
                )
            else:
                user_mask_2d_pad=None

            out_2d_padded = fused_self_attn_for_SD_small_head_size(
                q_ref=q_2d_pad,
                k_ref=k_2d_pad,
                v_ref=v_2d_pad,
                use_causal_mask=True,
                mixed_precision=True,
                mask_ref=user_mask_2d_pad
            )
            # slice => (seq_q,d_head)
            attn_out[b,h_,:seq_q] = out_2d_padded[:seq_q]

    return attn_out


def test_compute_for_token_gen():
    device = xm.xla_device()
    torch.manual_seed(0)

    B,H = 2,3
    seq_prev,seq_active,seq_q = 4,2,2
    d_head=64

    Q = torch.randn(B,H,seq_q,d_head,dtype=torch.float32,device=device)
    K_prev = torch.randn(B,H,seq_prev,d_head,dtype=torch.float32,device=device)
    V_prev = torch.randn(B,H,seq_prev,d_head,dtype=torch.float32,device=device)
    K_new = torch.randn(B,H,seq_active,d_head,dtype=torch.float32,device=device)
    V_new = torch.randn(B,H,seq_active,d_head,dtype=torch.float32,device=device)
    bool_mask = (torch.rand(B,H,seq_q,seq_prev+seq_active,device=device)>0.4)

    past_key_value=(K_prev,V_prev)
    position_ids = torch.zeros((B,seq_q),device=device,dtype=torch.long)
    active_mask = torch.empty(0,device=device)

    out_nki = compute_for_token_gen(
        self=None,
        Q=Q,
        K=K_new,
        V=V_new,
        position_ids=position_ids,
        past_key_value=past_key_value,
        attention_mask=bool_mask,
        active_mask=active_mask
    )
    print("[TEST] out_nki shape =", out_nki.shape)
    print("[TEST] out_nki[0,0,0,:8].cpu() =>", out_nki[0,0,0,:8].cpu().numpy())

if __name__=="__main__":
    test_compute_for_token_gen()