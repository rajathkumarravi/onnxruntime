/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qkv_quant_common.h

Abstract:

    Shared inline helpers for the quantized KV-cache GEMM kernel implementations.
    These utilities are used by the scalar reference (qkv_quant.cpp) and
    all SIMD-optimized backends (AVX2, AVX512-VNNI, NEON).

--*/

#pragma once

#include "mlas_qkv_quant.h"

#include <vector>

namespace MlasKVQuantInternal {

constexpr int kInt4Bias = 8;

inline bool
IsInt4Mode(MLAS_KV_QUANT_TYPE qt)
{
    return qt == MLAS_KV_QUANT_TYPE::S4_PerTensor ||
           qt == MLAS_KV_QUANT_TYPE::S4_PerChannel ||
           qt == MLAS_KV_QUANT_TYPE::S4_PerChannel_Gemm;
}

inline bool
IsPerChannelMode(MLAS_KV_QUANT_TYPE qt)
{
    return qt == MLAS_KV_QUANT_TYPE::S8_PerChannel ||
           qt == MLAS_KV_QUANT_TYPE::S4_PerChannel ||
           qt == MLAS_KV_QUANT_TYPE::F8E4M3FN_PerChannel ||
           qt == MLAS_KV_QUANT_TYPE::F8E5M2_PerChannel ||
           qt == MLAS_KV_QUANT_TYPE::S8_PerChannel_Gemm ||
           qt == MLAS_KV_QUANT_TYPE::S4_PerChannel_Gemm ||
           qt == MLAS_KV_QUANT_TYPE::S8_PerChannel_Native;
}

inline bool
IsFp8Mode(MLAS_KV_QUANT_TYPE qt)
{
    return qt == MLAS_KV_QUANT_TYPE::F8E4M3FN_PerChannel ||
           qt == MLAS_KV_QUANT_TYPE::F8E5M2_PerChannel;
}

inline bool
UsesDecodedGemm(MLAS_KV_QUANT_TYPE qt)
{
    return IsFp8Mode(qt) || qt == MLAS_KV_QUANT_TYPE::S8_PerChannel_Gemm ||
           qt == MLAS_KV_QUANT_TYPE::S4_PerChannel_Gemm;
}

inline bool
UsesNativeInt8Gemm(MLAS_KV_QUANT_TYPE qt)
{
    return qt == MLAS_KV_QUANT_TYPE::S8_PerChannel_Native;
}

struct NativeInt8Workspace {
    std::vector<int8_t> query;
    std::vector<float> query_scales;
    std::vector<float> float_row;
    std::vector<uint8_t> transposed_key;
    std::vector<uint8_t> probabilities;
    std::vector<float> probability_scales;
    std::vector<int32_t> products;
    size_t query_rows = 0;
    size_t query_columns = 0;
};

void PrepareNativeInt8Query(
    size_t M, size_t K, const float* A, size_t lda, const float* Scales, NativeInt8Workspace& Workspace);
void PrepareNativeInt8Query(
    size_t M, size_t K, const MLAS_FP16* A, size_t lda, const float* Scales, NativeInt8Workspace& Workspace);
void NativeInt8QKGemm(
    size_t M, size_t N, size_t K, float Alpha, const void* B, float* C, size_t ldc,
    NativeInt8Workspace& Workspace, MLAS_THREADPOOL* ThreadPool = nullptr);
void NativeInt8SVGemm(
    size_t M, size_t N, size_t K, const float* A, size_t lda, const void* B, const float* Scales,
    float* C, size_t ldc, float Beta, NativeInt8Workspace& Workspace, MLAS_THREADPOOL* ThreadPool = nullptr);

}  // namespace MlasKVQuantInternal
