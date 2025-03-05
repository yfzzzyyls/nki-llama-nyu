import copy
import sys
import gc
import logging
import math
import numpy as np
import time
from typing import List, Optional, Tuple, Type
import torch
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import os
from torch_xla.core import xla_model as xm

# os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
# os.environ["NEURON_CC_FLAGS"]= " --disable-dge "

@nki.jit
def nki_matmul_fully_optimized_spmd_(
    lhsT,
    rhs,
    TILES_IN_BLOCK_M=1,
    TILES_IN_BLOCK_N=16,
    TILES_IN_BLOCK_K=16,
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

def performance_test(device = xm.xla_device()):    
    x = torch.rand((128, 128), dtype=torch.bfloat16, device=device)
    weight = torch.rand((128, 512), dtype=torch.bfloat16, device=device)
    x_T = x.T
    nki_start = time.time()
    nki_out = nki_matmul_fully_optimized_spmd(x_T, weight, 1, 1, 1, 2)
    nki_end = time.time()
    nki_latency = nki_end - nki_start

    # device = torch.device("cpu")
    x_torch = torch.rand((128, 128), dtype=torch.bfloat16, device=device)
    weight_torch = torch.rand((128, 512), dtype=torch.bfloat16, device=device)
    torch_start = time.time()
    torch_out = torch.matmul(x_torch, weight_torch)
    torch_end = time.time()
    torch_latency = torch_end - torch_start
    print("nki shape, nki latency:", nki_out.size(), nki_latency)
    print("torch shape, torch latency:", torch_out.size(), torch_latency)


def test_matmul_SPMD(device = xm.xla_device()):
    x, gate_weight, up_weight = prepare_matmul_SPMD_input(device)

    B, N, C = x.shape   # For example, B * N = 32, C = 2048
    M_orig = B * N      # Original M dimension (32)

    # We need M to be a multiple of 256.
    BLOCK_M = 256
    M_pad = math.ceil(M_orig / BLOCK_M) * BLOCK_M  # Next multiple of 256

    # Flatten x to 2D: [B*N, C]
    x_flat = x.view(M_orig, C)

    # Pad x_flat along dimension 0 if needed.
    if M_pad > M_orig:
        pad_rows = M_pad - M_orig
        pad = torch.zeros(pad_rows, C, dtype=x.dtype, device=x.device)
        x_flat_padded = torch.cat([x_flat, pad], dim=0)
    else:
        x_flat_padded = x_flat
    
    MDIM = x_flat_padded.shape[0]
    NDIM = gate_weight.shape[0]
    KDIM = gate_weight.shape[1]

    Mtile = MDIM // 128 // 2 #### 2 for 2 cores / device
    Ntile = NDIM // 512
    Ktile = KDIM // 128 

    output_padded = nki_matmul_fully_optimized_spmd_[nl.nc(2)](x_flat_padded.T, gate_weight.T, Mtile, Ntile, Ktile, 2)
    output_flat = output_padded[:M_orig, :]
    gate_proj_output = output_flat.view(B, N, -1)

    check_matmul_match(x, gate_weight.T, gate_proj_output)

    return gate_proj_output

def check_result_match(torch_output, nki_output):
    # Ensure both outputs are on the same device (if necessary, move to CPU)
    cpu = torch.device('cpu')
    nki_output_cpu = nki_output.to(device=cpu)
    torch_output_cpu = torch_output.cpu()
    print(f"torch_matmul_output_shape={torch_output.shape}") # supposed to be 2048x4096
    print(f"torch_matmul_output_type={torch_output.type}") # supposed to be 2048x4096

    # Compare using torch.allclose with a tolerance (adjust atol/rtol as needed)
    if torch.allclose(torch_output_cpu, nki_output_cpu, atol=1e-5):
        print("The results match!")
    else:
        print("There is a mismatch between the outputs.")

def check_matmul_match(lsh, rsh, nki_output):
    cpu = torch.device('cpu')
    nki_output_cpu = nki_output.to(device=cpu)
     
    # Compute the expected result using torch.matmul (or any other reference method)
    torch_output = torch.matmul(lsh, rsh)

    # Ensure both outputs are on the same device (if necessary, move to CPU)
    torch_output_cpu = torch_output.cpu()

    # Compare using torch.allclose with a tolerance (adjust atol/rtol as needed)
    if torch.allclose(torch_output_cpu, nki_output_cpu, atol=1e-5):
        print("The results match!")
    else:
        print("There is a mismatch between the outputs.")

def prepare_mlp_gating_input(device = xm.xla_device()):
    # input
    x = torch.rand((1, 32, 2048), dtype=torch.bfloat16, device=device)
    # weights
    gate_weight = torch.rand((8192, 2048), dtype=torch.bfloat16, device=device)
    up_weight = torch.rand((8192, 2048), dtype=torch.bfloat16, device=device)
    down_weight = torch.rand((8192, 2048), dtype=torch.bfloat16, device=device)

    return x, gate_weight, up_weight, down_weight

def prepare_matmul_SPMD_input(device = xm.xla_device()):
    # input
    x = torch.rand((1, 32, 2048), dtype=torch.bfloat16, device=device)
    # weights
    gate_weight = torch.rand((8192, 2048), dtype=torch.bfloat16, device=device)
    up_weight = torch.rand((8192, 2048), dtype=torch.bfloat16, device=device)
    return x, gate_weight, up_weight


def main():
    # use Trn1 instance
    device = xm.xla_device()

    # test spmd optimized matmul
    spmd_matmul_result = test_matmul_SPMD(device)
    print(f"fully_optimized_output_shape={spmd_matmul_result.size()}")

if __name__ == "__main__":
    main()




 