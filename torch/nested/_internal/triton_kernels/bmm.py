"""
Group GEMM
============================
This group gemm kernel launches a fixed number of CTA to compute a group
of gemms. The scheduling is static and we do it on device.
"""

# Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch

import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 84,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 128,
            'BLOCK_SIZE_N': 128,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 84,
        }),
        triton.Config({
            'BLOCK_SIZE_M': 64,
            'BLOCK_SIZE_N': 64,
            'BLOCK_SIZE_K': 32,
            'NUM_SM': 128,
        }),
    ],
    key=['group_size'],
)
@triton.jit
def grouped_matmul_kernel(
    b_ptr_base,
    group_size,
    gn,
    gk,
    ldb,
    a_offsets_ptr,
    a_ptr_base,
    lda,
    ldc,
    c_ptr_base,
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tile_idx = tl.program_id(0)
    batch_id = tl.program_id(1)

    # get the gemm size of the current problem
    a_offset_0 = tl.load(a_offsets_ptr + batch_id)
    a_offset_1 = tl.load(a_offsets_ptr + batch_id + 1)
    gm = a_offset_1 - a_offset_0

    a_ptr = a_ptr_base + a_offset_0 * lda
    b_ptr = b_ptr_base + batch_id * gk * gn
    c_ptr = c_ptr_base + a_offset_0 * ldc

    num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
    num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
    num_tiles = num_m_tiles * num_n_tiles
    last_problem_end = num_tiles

    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # iterate through the tiles in the current gemm problem
    while tile_idx < last_problem_end:
        # tl.device_print("tile_idx: ", tile_idx)
        # pick up a tile from the current gemm problem
        # figure out tile coordinates
        tile_m_idx = tile_idx // num_n_tiles
        tile_n_idx = tile_idx % num_n_tiles

        # do regular gemm here
        offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        # tl.device_print("offs_am: ", offs_am)
        offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        a_ptrs = a_ptr + (offs_am[:, None] % gm) * lda + offs_k[None, :]
        b_ptrs = b_ptr + offs_k[:, None] * ldb + (offs_bn[None, :] % gn)
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for kk in range(0, tl.cdiv(gk, BLOCK_SIZE_K)):
            # # # hint to Triton compiler to do proper loop pipelining
            # tl.multiple_of(a_ptrs, [16, 16])
            # tl.multiple_of(b_ptrs, [16, 16])
            # assume full tile for now
            a = tl.load(a_ptrs, mask=offs_k[None, :] < gk - kk * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < gk - kk * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, b, accumulator, "ieee")
            a_ptrs += BLOCK_SIZE_K
            b_ptrs += BLOCK_SIZE_K * ldb
        c = accumulator.to(tl.float32)

        # offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        # offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        c_ptrs = c_ptr + ldc * offs_am[:, None] + offs_bn[None, :]

        c_mask = (offs_am[:, None] < gm) & (offs_bn[None, :] < gn)

        # tl.device_print("c_mask: ", c_mask)

        # assumes full tile for now
        tl.store(c_ptrs, c, mask=c_mask)

        # go to the next tile by advancing NUM_SM
        tile_idx += NUM_SM

def group_gemm_fn(tensor_a, tensor_b):
    assert tensor_a.is_nested
    assert not tensor_b.is_nested
    assert tensor_a.size(0) == tensor_b.size(0)
    group_size = tensor_a.size(0)

    assert tensor_b.is_contiguous()

    a_values = tensor_a.values()
    a_offsets = tensor_a.offsets().to(torch.int32)

    assert a_values.dim() == 2

    B, K, N = tensor_b.shape

    c_values = a_values.new_empty((a_values.size(0), N))
    c_offsets = a_offsets

    # we use a fixed number of CTA, and it's auto-tunable
    grid = lambda META: (META['NUM_SM'], group_size)
    # print("a_offsets: ", a_offsets)
    # grid = (128, group_size)
    # print("grid: ", grid)
    # print("tensor_b.size(): ", tensor_b.size())
    grouped_matmul_kernel[grid](
        tensor_b,
        group_size,
        N,
        K,
        tensor_b.stride(1),
        a_offsets,
        a_values,
        a_values.stride(0),
        c_values.stride(0),
        c_values,
        # BLOCK_SIZE_M=128,
        # BLOCK_SIZE_N=128,
        # BLOCK_SIZE_K=128,
        # NUM_SM=128,
        # num_stages=1,
    )

    return c_values