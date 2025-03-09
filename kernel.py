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

    # 1) Get their shapes
    B1, L1, E1 = lhs.shape
    B2, L2, E2 = rhs.shape

    # 2) (Optional) Assert shapes match how you expect
    #    The question is: "Is B always the same for both matrices?"
    #    If your code depends on them matching, you can assert that:
    assert B1 == B2, f"B mismatch: {B1} != {B2}"
    assert L1 == L2, f"L mismatch: {L1} != {L2}"
    assert E1 == E2, f"E mismatch: {E1} != {E2}"

    # 3) Flatten each from 3D -> 2D
    lhs_2d = lhs.view(B1 * L1, E1)
    rhs_2d = rhs.view(B2 * L2, E2)

    # 4) Run the NKI SiLU + multiply kernel on 2D
    print("lhs_2d.shape =", lhs_2d.shape)
    print("rhs_2d.shape =", rhs_2d.shape)
    nki_actfn_output_2d = nki_silu_mul(lhs_2d, rhs_2d)

    # 5) Un-flatten back to 3D
    nki_actfn_output_3d = nki_actfn_output_2d.view(B1, L1, E1)

    # 6) For reference, compute the default PyTorch-based result in 3D
    default_actfn = get_actfn("silu")  # e.g. F.silu
    default_actfn_output_3d = default_actfn(lhs) * rhs

    # 7) Compare the two outputs
    check_2matrices_match(default_actfn_output_3d, nki_actfn_output_3d)


@nki.jit
def nki_silu_mul(lhs_tensor, rhs_tensor):
    # Expect 2D shape (H, W)
    orig_shape = lhs_tensor.shape
    H, W = orig_shape

    lhs_nl = nl.load(lhs_tensor)
    rhs_nl = nl.load(rhs_tensor)

    lhs_silu = nl.silu(lhs_nl)
    out_nl   = nl.multiply(lhs_silu, rhs_nl)

    out_tensor = nl.ndarray(orig_shape, dtype=lhs_tensor.dtype, buffer=nl.shared_hbm)
    nl.store(out_tensor, value=out_nl)
    return out_tensor

def generate_2_matrices(device = xm.xla_device()):
    x = torch.rand((1, 32, 4096), dtype=torch.bfloat16, device=device)
    y = torch.rand((1, 32, 4096), dtype=torch.bfloat16, device=device)
    return x, y

def main():
    # use Trn1 instance
    device = xm.xla_device()
    # # test SPMD matmul
    # nki_output = test_mlp_gating_fully_optimized_matmul(device)

    # test nki activation fucntion
    test_nki_silu_mul(device)

    # nki_output, lsh, rsh = test_matmul_singletile()
    # nki_output, lsh, rsh = test_block_free()
    # check_match(lsh, rsh, nki_output)
    # performance_test()

if __name__ == "__main__":
    main()
  


 