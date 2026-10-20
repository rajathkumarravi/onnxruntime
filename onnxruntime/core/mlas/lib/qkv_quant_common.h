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

#include "../inc/mlas_qkv_quant.h"

#include <vector>

namespace MlasKVQuantInternal {

constexpr int kInt4Bias = 8;

inline bool
IsInt4Mode(MLAS_KV_QUANT_TYPE qt)
{
    return qt == MLAS_KV_QUANT_TYPE::S4_PerTensor ||
           qt == MLAS_KV_QUANT_TYPE::S4_PerChannel ||
           qt == MLAS_KV_QUANT_TYPE::S4_PerChannel_Gemm ||
           qt == MLAS_KV_QUANT_TYPE::S4_PerChannel_Native;
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
           qt == MLAS_KV_QUANT_TYPE::S8_PerChannel_Native ||
           qt == MLAS_KV_QUANT_TYPE::S4_PerChannel_Native ||
           qt == MLAS_KV_QUANT_TYPE::F8E4M3FN_PerChannel_Fp16 ||
           qt == MLAS_KV_QUANT_TYPE::F8E5M2_PerChannel_Fp16;
}

inline bool
IsFp8E4M3Mode(MLAS_KV_QUANT_TYPE qt)
{
    return qt == MLAS_KV_QUANT_TYPE::F8E4M3FN_PerChannel ||
           qt == MLAS_KV_QUANT_TYPE::F8E4M3FN_PerChannel_Fp16;
}

inline bool
IsFp8Mode(MLAS_KV_QUANT_TYPE qt)
{
    return qt == MLAS_KV_QUANT_TYPE::F8E4M3FN_PerChannel ||
           qt == MLAS_KV_QUANT_TYPE::F8E5M2_PerChannel ||
           qt == MLAS_KV_QUANT_TYPE::F8E4M3FN_PerChannel_Fp16 ||
           qt == MLAS_KV_QUANT_TYPE::F8E5M2_PerChannel_Fp16;
}

inline bool
UsesFp8HalfGemm(MLAS_KV_QUANT_TYPE qt)
{
    return qt == MLAS_KV_QUANT_TYPE::F8E4M3FN_PerChannel_Fp16 ||
           qt == MLAS_KV_QUANT_TYPE::F8E5M2_PerChannel_Fp16;
}

inline bool
UsesDecodedGemm(MLAS_KV_QUANT_TYPE qt)
{
    return (IsFp8Mode(qt) && !UsesFp8HalfGemm(qt)) ||
           qt == MLAS_KV_QUANT_TYPE::S8_PerChannel_Gemm ||
           qt == MLAS_KV_QUANT_TYPE::S4_PerChannel_Gemm;
}

inline bool
UsesNativeInt8Gemm(MLAS_KV_QUANT_TYPE qt)
{
    return qt == MLAS_KV_QUANT_TYPE::S8_PerChannel_Native;
}

inline bool
UsesNativeInt4Gemm(MLAS_KV_QUANT_TYPE qt)
{
    return qt == MLAS_KV_QUANT_TYPE::S4_PerChannel_Native;
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

struct NativeInt4Workspace {
    NativeInt8Workspace native_int8;
    std::vector<int8_t> unpacked;
    std::vector<float> query;
    std::vector<uint8_t> blocked;
    std::vector<uint8_t> packed;
    std::vector<uint8_t> gemm_workspace;
    std::vector<float> scales;
    std::vector<float> products;
    size_t query_rows = 0;
    size_t query_columns = 0;
};

void PrepareNativeInt4Query(
    size_t M, size_t K, const float* A, size_t lda, const float* Scales, NativeInt4Workspace& Workspace);
void PrepareNativeInt4Query(
    size_t M, size_t K, const MLAS_FP16* A, size_t lda, const float* Scales, NativeInt4Workspace& Workspace);
void NativeInt4QKGemm(
    size_t M, size_t N, size_t K, float Alpha, const void* B, float* C, size_t ldc,
    NativeInt4Workspace& Workspace, MLAS_THREADPOOL* ThreadPool = nullptr);
void NativeInt4SVGemm(
    size_t M, size_t N, size_t K, const float* A, size_t lda, const void* B, const float* Scales,
    float* C, size_t ldc, float Beta, NativeInt4Workspace& Workspace, MLAS_THREADPOOL* ThreadPool = nullptr);

struct Fp8HalfWorkspace {
    std::vector<MLAS_FP16> query;
    std::vector<MLAS_FP16> decoded;
    std::vector<MLAS_FP16> probabilities;
    std::vector<MLAS_FP16> products;
    const MLAS_FP16* prepared_query = nullptr;
    size_t query_stride = 0;
    size_t query_rows = 0;
    size_t query_columns = 0;
};

struct DecodedGemmWorkspace {
    std::vector<float> query;
    std::vector<float> decoded;
    const float* prepared_query = nullptr;
    size_t query_stride = 0;
    size_t query_rows = 0;
    size_t query_columns = 0;
};

void PrepareDecodedQuery(
    size_t M, size_t K, const float* A, size_t lda, DecodedGemmWorkspace& Workspace);
void PrepareDecodedQuery(
    size_t M, size_t K, const MLAS_FP16* A, size_t lda, DecodedGemmWorkspace& Workspace);
void DecodedQKGemm(
    size_t M, size_t N, size_t K, float Alpha, const void* B, MLAS_KV_QUANT_TYPE QuantType,
    const float* Scales, float* C, size_t ldc, DecodedGemmWorkspace& Workspace,
    MLAS_THREADPOOL* ThreadPool = nullptr);
void DecodedSVGemm(
    size_t M, size_t N, size_t K, const float* A, size_t lda, const void* B,
    MLAS_KV_QUANT_TYPE QuantType, const float* Scales, float* C, size_t ldc, float Beta,
    DecodedGemmWorkspace& Workspace, MLAS_THREADPOOL* ThreadPool = nullptr);

void PrepareFp8HalfQuery(
    size_t M, size_t K, const float* A, size_t lda, Fp8HalfWorkspace& Workspace);
void PrepareFp8HalfQuery(
    size_t M, size_t K, const MLAS_FP16* A, size_t lda, Fp8HalfWorkspace& Workspace);
void Fp8HalfQKGemm(
    size_t M, size_t N, size_t K, float Alpha, const void* B, MLAS_KV_QUANT_TYPE QuantType,
    const float* Scales, float* C, size_t ldc, Fp8HalfWorkspace& Workspace,
    MLAS_THREADPOOL* ThreadPool = nullptr);
void Fp8HalfSVGemm(
    size_t M, size_t N, size_t K, const float* A, size_t lda, const void* B,
    MLAS_KV_QUANT_TYPE QuantType, const float* Scales, float* C, size_t ldc, float Beta,
    Fp8HalfWorkspace& Workspace, MLAS_THREADPOOL* ThreadPool = nullptr);

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