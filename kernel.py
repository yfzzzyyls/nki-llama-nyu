import torch
from neuronxcc import nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import math
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
def nki_matmul_block_free_dimension_(lhsT, rhs):
    # print("K dim, k_ dim", lhsT.shape, rhs.shape)

    K, M = lhsT.shape
    K_, N = rhs.shape
  
    assert K == K_, "lhsT and rhs must have the same contraction dimension"
    result = nl.ndarray((M, N), dtype=lhsT.dtype, buffer=nl.shared_hbm)

    TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
    TILE_K = nl.tile_size.pmax  # 128
    TILE_N = nl.tile_size.gemm_moving_fmax  # 512

    # Define the indices (shape) of the tiles
    i_lhsT = nl.mgrid[0:TILE_K, 0:TILE_M]
    i_rhs = nl.mgrid[0:TILE_K, 0:TILE_N]
    i_res = nl.mgrid[0:TILE_M, 0:TILE_N]

    # Configuring the blocking size for the free dimensions
    TILES_IN_BLOCK_M = 2
    TILES_IN_BLOCK_N = 2

    BLOCK_M = TILE_M * TILES_IN_BLOCK_M  # 256
    BLOCK_N = TILE_N * TILES_IN_BLOCK_N  # 1024

    # the size has to be multiple of block size
    assert M % BLOCK_M == 0
    assert N % BLOCK_N == 0

    # Loop over blocks over the M dimension
    for m in nl.affine_range(M // BLOCK_M):
        # Load TILES_IN_BLOCK_M columns tiles from lhsT
        lhsT_tiles = nl.ndarray(
            (TILES_IN_BLOCK_M, K // TILE_K, nl.par_dim(TILE_K), TILE_M),
            dtype=lhsT.dtype,
            buffer=nl.sbuf)
        for bm in nl.affine_range(TILES_IN_BLOCK_M):
            for k in nl.affine_range(K // TILE_K):
                lhsT_tiles[bm, k, i_lhsT.p, i_lhsT.x] = nl.load(
                    lhsT[k * TILE_K + i_lhsT.p,
                        (m * TILES_IN_BLOCK_M + bm) * TILE_M + i_lhsT.x])

        for n in nl.affine_range(N // BLOCK_N):
            # Load TILES_IN_BLOCK_N columns from rhs
            rhs_tiles = nl.ndarray(
                (TILES_IN_BLOCK_N, K // TILE_K, nl.par_dim(TILE_K), TILE_N),
                dtype=rhs.dtype,
                buffer=nl.sbuf)
            for bn in nl.affine_range(TILES_IN_BLOCK_N):
                for k in nl.affine_range(K // TILE_K):
                    rhs_tiles[bn, k, i_rhs.p, i_rhs.x] = nl.load(
                        rhs[k * TILE_K + i_rhs.p,
                            (n * TILES_IN_BLOCK_N + bn) * TILE_N + i_rhs.x])

            for bm in nl.affine_range(TILES_IN_BLOCK_M):
                for bn in nl.affine_range(TILES_IN_BLOCK_N):
                # Allocate a tensor in PSUM
                    res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
                    for k in nl.affine_range(K // TILE_K):
                        # Accumulate partial-sums into PSUM
                        res_psum += nl.matmul(lhsT_tiles[bm, k, i_lhsT.p, i_lhsT.x],
                                            rhs_tiles[bn, k, i_rhs.p, i_rhs.x],
                                            transpose_x=True)

                    # Copy the result from PSUM back to SBUF, and cast to expected output data-type
                    res_sb = nl.copy(res_psum, dtype=result.dtype)
                    nl.store(result[(m * TILES_IN_BLOCK_M + bm) * TILE_M + i_res.p,
                                    (n * TILES_IN_BLOCK_N + bn) * TILE_N + i_res.x],
                            value=res_sb)

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


def test_block_free(device = xm.xla_device()):
    # lshT: input
    x = torch.rand((1, 32, 2048), dtype=torch.bfloat16, device=device)

    # rsh: weight
    weight = torch.rand((8192, 2048), dtype=torch.bfloat16, device=device)

    B, N, C = x.shape   # For example, B * N = 32, C = 2048
    M_orig = B * N      # Original M dimension (32)

    # We need M to be a multiple of 128.
    BLOCK_M = 256
    M_pad = math.ceil(M_orig / BLOCK_M) * BLOCK_M  # Next multiple of 128

    # Flatten x to 2D: [B*N, C]
    x_flat = x.view(M_orig, C)

    # Pad x_flat along dimension 0 if needed.
    if M_pad > M_orig:
        pad_rows = M_pad - M_orig
        pad = torch.zeros(pad_rows, C, dtype=x.dtype, device=x.device)
        x_flat_padded = torch.cat([x_flat, pad], dim=0)
    else:
        x_flat_padded = x_flat

    print("input padded shape", x_flat_padded.T.size())
    print("weight transpose shape", weight.T.size())        
    output_padded = nki_matmul_block_free_dimension_(x_flat_padded, weight.T)
    output_flat = output_padded[:M_orig, :]
    gate_proj_output = output_flat.view(B, N, -1)

    print(f"matmul_output_shape={gate_proj_output.shape}") # supposed to be 2048x4096
    print(f"matmul_output_type={gate_proj_output.type}")

    return gate_proj_output, x, weight.T

def test_fully_optimized_matmul(device = xm.xla_device()):
    # lshT: input
    x = torch.rand((1, 32, 2048), dtype=torch.bfloat16, device=device)

    # rsh: weight
    weight = torch.rand((8192, 2048), dtype=torch.bfloat16, device=device)

    B, N, C = x.shape   # For example, B * N = 32, C = 2048
    M_orig = B * N      # Original M dimension (32)

    # We need M to be a multiple of 128.
    BLOCK_M = 256
    M_pad = math.ceil(M_orig / BLOCK_M) * BLOCK_M  # Next multiple of 128

    # Flatten x to 2D: [B*N, C]
    x_flat = x.view(M_orig, C)

    # Pad x_flat along dimension 0 if needed.
    if M_pad > M_orig:
        pad_rows = M_pad - M_orig
        pad = torch.zeros(pad_rows, C, dtype=x.dtype, device=x.device)
        x_flat_padded = torch.cat([x_flat, pad], dim=0)
    else:
        x_flat_padded = x_flat

    print("input padded shape", x_flat_padded.T.size())
    print("weight transpose shape", weight.T.size())        
    output_padded = nki_matmul_fully_optimized_(x_flat_padded.T, weight.T)
    output_flat = output_padded[:M_orig, :]
    gate_proj_output = output_flat.view(B, N, -1)

    print(f"fully_optimized_output_shape={gate_proj_output.shape}") # supposed to be 2048x4096

    return gate_proj_output, x, weight.T

def check_match(lsh, rsh, nki_output):
    cpu = torch.device('cpu')
    nki_output_cpu = nki_output.to(device=cpu)
     
    # Compute the expected result using torch.matmul (or any other reference method)
    torch_output = torch.matmul(lsh, rsh)

    # Ensure both outputs are on the same device (if necessary, move to CPU)
    torch_output_cpu = torch_output.cpu()
    print(f"torch_matmul_output_shape={torch_output.shape}") # supposed to be 2048x4096
    print(f"torch_matmul_output_type={torch_output.type}") # supposed to be 2048x4096

    # Compare using torch.allclose with a tolerance (adjust atol/rtol as needed)
    if torch.allclose(torch_output_cpu, nki_output_cpu, atol=1e-5):
        print("The results match!")
    else:
        print("There is a mismatch between the outputs.")

def main():
    # device = xm.xla_device() # device name
    # nki_output, lsh, rsh = test_matmul_singletile()
    nki_output, lsh, rsh = test_fully_optimized_matmul()
    # nki_output, lsh, rsh = test_block_free()
    check_match(lsh, rsh, nki_output)

if __name__ == "__main__":
    main()
  


 