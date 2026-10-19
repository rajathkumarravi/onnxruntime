/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qkv_quant.cpp

Abstract:

    Portable scalar reference implementation of the symmetric INT4 / INT8
    quantized KV-cache GEMM API declared in mlas_qkv_quant.h. This file provides
    a correct scalar fallback; SIMD-optimized backends (AVX2, AVX512-VNNI, NEON)
    are dispatched at runtime via the platform dispatch table.

    See mlas_qkv_quant.h for the packing, scaling, and layout contract.

--*/

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

#include "core/common/float8.h"
#include "core/common/safeint.h"
#include "mlas_qkv_quant.h"
#include "mlasi.h"
#include "qkv_quant_common.h"
#include "qkv_quant_kernel.h"

using namespace MlasKVQuantInternal;

namespace {

constexpr int kInt4Min = -8;
constexpr int kInt4Max = 7;
constexpr int kInt8Min = -128;
constexpr int kInt8Max = 127;

// Round-to-nearest-even via rintf, matching the CUDA QDQ implementation.
inline int8_t
QuantizeInt8(float x, float inv_scale)
{
    const float q = std::rintf(x * inv_scale);
    const int qi = static_cast<int>(std::max(static_cast<float>(kInt8Min),
                                             std::min(static_cast<float>(kInt8Max), q)));
    return static_cast<int8_t>(qi);
}

inline int
QuantizeInt4Nibble(float x, float inv_scale)
{
    const float q = std::rintf(x * inv_scale);
    const int qi = static_cast<int>(std::max(static_cast<float>(kInt4Min),
                                             std::min(static_cast<float>(kInt4Max), q)));
    return qi;
}

inline float
SafeInvScale(float scale)
{
    // A zero scale (typically meaning "no data yet") is treated as 1.0 so the
    // quantize step degenerates to a clamp/round rather than producing NaN.
    return (scale != 0.0f) ? (1.0f / scale) : 1.0f;
}

template <typename T>
void
QuantizeRowFp8(const T* src, uint8_t* dst, size_t cols, MLAS_KV_QUANT_TYPE type, const float* scales)
{
#if !defined(DISABLE_FLOAT8_TYPES)
    for (size_t c = 0; c < cols; ++c) {
        const float value = static_cast<float>(src[c]) * SafeInvScale(scales[c]);
        dst[c] = type == MLAS_KV_QUANT_TYPE::F8E4M3FN_PerChannel
                     ? onnxruntime::Float8E4M3FN(value, true).val
                     : onnxruntime::Float8E5M2(value, true).val;
    }
#else
    ORT_UNUSED_PARAMETER(src);
    ORT_UNUSED_PARAMETER(dst);
    ORT_UNUSED_PARAMETER(cols);
    ORT_UNUSED_PARAMETER(type);
    ORT_UNUSED_PARAMETER(scales);
    ORT_THROW("FP8 KV support is disabled in this build");
#endif
}

const std::array<float, 256>&
Fp8Lookup(MLAS_KV_QUANT_TYPE type)
{
#if !defined(DISABLE_FLOAT8_TYPES)
    static const auto e4m3 = [] {
        std::array<float, 256> values{};
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = onnxruntime::Float8E4M3FN(
                            static_cast<uint8_t>(i), onnxruntime::Float8E4M3FN::FromBits()
            )
                            .ToFloat();
        }
        return values;
    }();
    static const auto e5m2 = [] {
        std::array<float, 256> values{};
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = onnxruntime::Float8E5M2(
                            static_cast<uint8_t>(i), onnxruntime::Float8E5M2::FromBits()
            )
                            .ToFloat();
        }
        return values;
    }();
    return type == MLAS_KV_QUANT_TYPE::F8E4M3FN_PerChannel ? e4m3 : e5m2;
#else
    ORT_UNUSED_PARAMETER(type);
    ORT_THROW("FP8 KV support is disabled in this build");
#endif
}

void
DequantizeRowFp8(const uint8_t* src, float* dst, size_t cols, MLAS_KV_QUANT_TYPE type, const float* scales)
{
    const auto& values = Fp8Lookup(type);
    for (size_t c = 0; c < cols; ++c) {
        dst[c] = values[src[c]] * scales[c];
    }
}

inline float
DequantInt8(int8_t q, float scale)
{
    return static_cast<float>(q) * scale;
}

// Decode a single nibble from a packed byte. `col` is the unpacked column
// index; even columns occupy the low nibble, odd columns the high nibble.
inline float
DequantInt4FromByte(uint8_t packed, size_t col, float scale)
{
    const int nibble = (col & 1U) == 0 ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
    return static_cast<float>(nibble - kInt4Bias) * scale;
}

// Pack one row of FP32 source into INT8 destination.
void
QuantizeRowInt8(
    const float* src,
    int8_t* dst,
    size_t cols,
    bool per_channel,
    const float* scales)
{
    if (per_channel) {
        for (size_t c = 0; c < cols; ++c) {
            dst[c] = QuantizeInt8(src[c], SafeInvScale(scales[c]));
        }
    } else {
        const float inv_scale = SafeInvScale(scales[0]);
        for (size_t c = 0; c < cols; ++c) {
            dst[c] = QuantizeInt8(src[c], inv_scale);
        }
    }
}

// Pack one row of FP32 source into INT4 destination using the +8-biased
// two-per-byte convention. If `cols` is odd, the trailing high nibble is set
// to 0 (which decodes to -8 * scale; matches the CUDA "pair with q1 = 0"
// convention and is fine because such columns are out of range for the
// consumer).
void
QuantizeRowInt4(
    const float* src,
    uint8_t* dst,
    size_t cols,
    bool per_channel,
    const float* scales)
{
    const size_t out_bytes = (cols + 1) / 2;
    for (size_t b = 0; b < out_bytes; ++b) {
        const size_t c0 = 2 * b;
        const size_t c1 = c0 + 1;

        const float inv0 = per_channel ? SafeInvScale(scales[c0])
                                       : SafeInvScale(scales[0]);
        const int q0 = QuantizeInt4Nibble(src[c0], inv0);
        int q1 = 0;
        if (c1 < cols) {
            const float inv1 = per_channel ? SafeInvScale(scales[c1])
                                           : SafeInvScale(scales[0]);
            q1 = QuantizeInt4Nibble(src[c1], inv1);
        }
        dst[b] = static_cast<uint8_t>(
            (((q0 + kInt4Bias) & 0x0F)) |
            ((((q1 + kInt4Bias) & 0x0F)) << 4));
    }
}

void
QuantizeRowInt8Fp16(
    const MLAS_FP16* src,
    int8_t* dst,
    size_t cols,
    bool per_channel,
    const float* scales)
{
    if (per_channel) {
        for (size_t c = 0; c < cols; ++c) {
            dst[c] = QuantizeInt8(static_cast<float>(src[c]), SafeInvScale(scales[c]));
        }
    } else {
        const float inv_scale = SafeInvScale(scales[0]);
        for (size_t c = 0; c < cols; ++c) {
            dst[c] = QuantizeInt8(static_cast<float>(src[c]), inv_scale);
        }
    }
}

void
QuantizeRowInt4Fp16(
    const MLAS_FP16* src,
    uint8_t* dst,
    size_t cols,
    bool per_channel,
    const float* scales)
{
    const size_t out_bytes = (cols + 1) / 2;
    for (size_t b = 0; b < out_bytes; ++b) {
        const size_t c0 = 2 * b;
        const size_t c1 = c0 + 1;
        const float inv0 = per_channel ? SafeInvScale(scales[c0])
                                       : SafeInvScale(scales[0]);
        const int q0 = QuantizeInt4Nibble(static_cast<float>(src[c0]), inv0);
        int q1 = 0;
        if (c1 < cols) {
            const float inv1 = per_channel ? SafeInvScale(scales[c1])
                                           : SafeInvScale(scales[0]);
            q1 = QuantizeInt4Nibble(static_cast<float>(src[c1]), inv1);
        }
        dst[b] = static_cast<uint8_t>(
            (((q0 + kInt4Bias) & 0x0F)) |
            ((((q1 + kInt4Bias) & 0x0F)) << 4));
    }
}

// Dequantize one row from INT8.
void
DequantizeRowInt8(
    const int8_t* src,
    float* dst,
    size_t cols,
    bool per_channel,
    const float* scales)
{
    if (per_channel) {
        for (size_t c = 0; c < cols; ++c) {
            dst[c] = DequantInt8(src[c], scales[c]);
        }
    } else {
        const float scale = scales[0];
        for (size_t c = 0; c < cols; ++c) {
            dst[c] = DequantInt8(src[c], scale);
        }
    }
}

// Dequantize one row from packed INT4.
void
DequantizeRowInt4(
    const uint8_t* src,
    float* dst,
    size_t cols,
    bool per_channel,
    const float* scales)
{
    for (size_t c = 0; c < cols; ++c) {
        const uint8_t packed = src[c / 2];
        const float scale = per_channel ? scales[c] : scales[0];
        dst[c] = DequantInt4FromByte(packed, c, scale);
    }
}

}  // namespace

namespace MlasKVQuantInternal {
namespace {

template <typename T>
T* WorkspaceBuffer(std::vector<T>& buffer, size_t count)
{
    // Keep the high-water size so alternating QK/SV shapes do not repeatedly zero-fill scratch.
    if (buffer.size() < count) {
        buffer.resize(count);
    }
    return buffer.data();
}

void ValidateNativeInt8Scales(const float* scales, size_t count)
{
    ORT_ENFORCE(MlasIsKVQuantGemmSupported(MLAS_KV_QUANT_TYPE::S8_PerChannel_Native),
                "Native INT8 KV attention requires ARM64 integer GEMM support");
    ORT_ENFORCE(scales != nullptr, "Native INT8 KV scales are required");
    for (size_t i = 0; i < count; ++i) {
        ORT_ENFORCE(std::isfinite(scales[i]) && scales[i] > 0.0f,
                    "Native INT8 KV scales must be finite and positive");
    }
}

float NativeInt8Scale(float peak, float limit)
{
    // Tiny online-softmax tiles must not underflow the quantization scale to zero.
    return peak == 0.0f ? 1.0f : std::max(peak / limit, std::numeric_limits<float>::min());
}

bool NativeInt8FindFiniteRange(const float* input, size_t count, float& minimum, float& maximum)
{
    minimum = std::numeric_limits<float>::max();
    maximum = std::numeric_limits<float>::lowest();
    bool finite = true;
    size_t i = 0;
#if defined(MLAS_TARGET_ARM64)
    float32x4_t min_values = vdupq_n_f32(minimum);
    float32x4_t max_values = vdupq_n_f32(maximum);
    uint32x4_t valid = vdupq_n_u32(UINT32_MAX);
    const uint32x4_t absolute_mask = vdupq_n_u32(0x7fffffff);
    const uint32x4_t infinity = vdupq_n_u32(0x7f800000);
    for (; i + 4 <= count; i += 4) {
        const float32x4_t values = vld1q_f32(input + i);
        min_values = vminq_f32(min_values, values);
        max_values = vmaxq_f32(max_values, values);
        const uint32x4_t bits = vandq_u32(vreinterpretq_u32_f32(values), absolute_mask);
        valid = vandq_u32(valid, vcltq_u32(bits, infinity));
    }
    minimum = vminvq_f32(min_values);
    maximum = vmaxvq_f32(max_values);
    finite = vminvq_u32(valid) == UINT32_MAX;
#endif
    for (; i < count; ++i) {
        finite &= std::isfinite(input[i]);
        minimum = std::min(minimum, input[i]);
        maximum = std::max(maximum, input[i]);
    }
    return finite;
}

template <typename T>
void PrepareNativeInt8QueryImpl(
    size_t M, size_t K, const T* A, size_t lda, const float* scales, NativeInt8Workspace& workspace)
{
    ValidateNativeInt8Scales(scales, K);
    ORT_ENFORCE(K > 0 && K <= static_cast<size_t>(std::numeric_limits<int32_t>::max()) / (128 * 128),
                "Native INT8 QK reduction is outside the INT32 accumulation range");
    ORT_ENFORCE(lda >= K, "Invalid native INT8 query row stride");
    workspace.query_rows = workspace.query_columns = 0;
    auto* query = WorkspaceBuffer(workspace.query, SafeInt<size_t>(M) * K);
    auto* row_scales = WorkspaceBuffer(workspace.query_scales, M);
    auto* row = WorkspaceBuffer(workspace.float_row, K);
    for (size_t m = 0; m < M; ++m) {
        if constexpr (std::is_same_v<T, float>) {
            std::copy_n(A + m * lda, K, row);
        } else {
            MlasConvertHalfToFloatBuffer(A + m * lda, row, K);
        }
        // Fold per-channel K scaling into Q before its per-row quantization.
        size_t k = 0;
#if defined(MLAS_TARGET_ARM64)
        for (; k + 4 <= K; k += 4) {
            vst1q_f32(row + k, vmulq_f32(vld1q_f32(row + k), vld1q_f32(scales + k)));
        }
#endif
        for (; k < K; ++k) {
            row[k] *= scales[k];
        }
        float minimum, maximum;
        ORT_ENFORCE(NativeInt8FindFiniteRange(row, K, minimum, maximum),
                    "Native INT8 attention requires finite scaled queries");
        row_scales[m] = NativeInt8Scale(std::max(std::abs(minimum), std::abs(maximum)), 127.0f);
        MlasQuantizeLinear(row, query + m * K, K, row_scales[m], int8_t{0});
    }
    workspace.query_rows = M;
    workspace.query_columns = K;
}

void NativeInt8Product(
    size_t M, size_t N, size_t K, const uint8_t* A, size_t lda, bool signed_a,
    const void* B, size_t ldb, NativeInt8Workspace& workspace, MLAS_THREADPOOL* thread_pool)
{
    MLAS_GEMM_QUANT_SHAPE_PARAMS shape;
    shape.M = M;
    shape.N = N;
    shape.K = K;
    shape.AIsSigned = signed_a;
    shape.BIsSigned = true;
    const uint8_t zero = 0;
    MLAS_GEMM_QUANT_DATA_PARAMS data;
    data.A = A;
    data.lda = lda;
    data.B = B;
    data.ldb = ldb;
    data.ZeroPointB = &zero;
    data.C = WorkspaceBuffer(workspace.products, SafeInt<size_t>(M) * N);
    data.ldc = N;
    MlasGemm(shape, data, thread_pool);
}

}  // namespace

void PrepareNativeInt8Query(
    size_t M, size_t K, const float* A, size_t lda, const float* Scales, NativeInt8Workspace& Workspace)
{
    PrepareNativeInt8QueryImpl(M, K, A, lda, Scales, Workspace);
}

void PrepareNativeInt8Query(
    size_t M, size_t K, const MLAS_FP16* A, size_t lda, const float* Scales, NativeInt8Workspace& Workspace)
{
    PrepareNativeInt8QueryImpl(M, K, A, lda, Scales, Workspace);
}

void NativeInt8QKGemm(
    size_t M, size_t N, size_t K, float Alpha, const void* B, float* C, size_t ldc,
    NativeInt8Workspace& Workspace, MLAS_THREADPOOL* ThreadPool)
{
    ORT_ENFORCE(Workspace.query_rows == M && Workspace.query_columns == K,
                "Native INT8 QK query tile was not prepared");
    auto* transposed = WorkspaceBuffer(Workspace.transposed_key, SafeInt<size_t>(N) * K);
    MlasTranspose(static_cast<const uint8_t*>(B), transposed, N, K, nullptr);
    NativeInt8Product(M, N, K, reinterpret_cast<const uint8_t*>(Workspace.query.data()), K, true,
                     transposed, N, Workspace, ThreadPool);
    for (size_t m = 0; m < M; ++m) {
        const float scale = Alpha * Workspace.query_scales[m];
        size_t n = 0;
#if defined(MLAS_TARGET_ARM64)
        for (; n + 4 <= N; n += 4) {
            const float32x4_t values = vcvtq_f32_s32(vld1q_s32(Workspace.products.data() + m * N + n));
            vst1q_f32(C + m * ldc + n, vmulq_n_f32(values, scale));
        }
#endif
        for (; n < N; ++n) {
            C[m * ldc + n] = static_cast<float>(Workspace.products[m * N + n]) * scale;
        }
    }
}

void NativeInt8SVGemm(
    size_t M, size_t N, size_t K, const float* A, size_t lda, const void* B, const float* Scales,
    float* C, size_t ldc, float Beta, NativeInt8Workspace& Workspace, MLAS_THREADPOOL* ThreadPool)
{
    ValidateNativeInt8Scales(Scales, N);
    ORT_ENFORCE(K > 0 && K <= static_cast<size_t>(std::numeric_limits<int32_t>::max()) / (255 * 128),
                "Native INT8 SV reduction is outside the INT32 accumulation range");
    auto* packed = WorkspaceBuffer(Workspace.probabilities, SafeInt<size_t>(M) * K);
    auto* row_scales = WorkspaceBuffer(Workspace.probability_scales, M);
    bool signed_a = false;
    for (size_t m = 0; m < M; ++m) {
        const float* row = A + m * lda;
        float minimum, maximum;
        ORT_ENFORCE(NativeInt8FindFiniteRange(row, K, minimum, maximum),
                    "Native INT8 attention requires finite probabilities");
        signed_a |= minimum < 0.0f;
        row_scales[m] = std::max(std::abs(minimum), std::abs(maximum));
    }
    for (size_t m = 0; m < M; ++m) {
        row_scales[m] = NativeInt8Scale(row_scales[m], signed_a ? 127.0f : 255.0f);
        if (signed_a) {
            MlasQuantizeLinear(A + m * lda, reinterpret_cast<int8_t*>(packed + m * K),
                               K, row_scales[m], int8_t{0});
        } else {
            MlasQuantizeLinear(A + m * lda, packed + m * K, K, row_scales[m], uint8_t{0});
        }
    }
    NativeInt8Product(M, N, K, packed, K, signed_a, B, N, Workspace, ThreadPool);
    for (size_t m = 0; m < M; ++m) {
        size_t n = 0;
#if defined(MLAS_TARGET_ARM64)
        for (; n + 4 <= N; n += 4) {
            const float32x4_t products = vcvtq_f32_s32(vld1q_s32(Workspace.products.data() + m * N + n));
            float32x4_t values = vmulq_f32(vmulq_n_f32(products, row_scales[m]), vld1q_f32(Scales + n));
            if (Beta != 0.0f) {
                values = vaddq_f32(vmulq_n_f32(vld1q_f32(C + m * ldc + n), Beta), values);
            }
            vst1q_f32(C + m * ldc + n, values);
        }
#endif
        for (; n < N; ++n) {
            const float value = static_cast<float>(Workspace.products[m * N + n]) *
                                row_scales[m] * Scales[n];
            C[m * ldc + n] = Beta == 0.0f ? value : Beta * C[m * ldc + n] + value;
        }
    }
}

}  // namespace MlasKVQuantInternal

bool
    MLASCALL
    MlasIsKVQuantGemmSupported([[maybe_unused]] MLAS_KV_QUANT_TYPE QuantType)
{
    if (UsesNativeInt8Gemm(QuantType)) {
#if defined(MLAS_TARGET_ARM64)
        return true;
#else
        return false;
#endif
    }
#if defined(DISABLE_FLOAT8_TYPES)
    if (IsFp8Mode(QuantType)) {
        return false;
    }
#endif
    // The portable reference path supports every mode on every platform.
    return true;
}

size_t
MLASCALL
MlasKVQuantPackedRowBytes(MLAS_KV_QUANT_TYPE QuantType, size_t Cols)
{
    return IsInt4Mode(QuantType) ? (Cols + 1) / 2 : Cols;
}

void
MLASCALL
MlasKVQuantize(
    const float* Src,
    void* Dst,
    size_t Rows,
    size_t Cols,
    size_t lda,
    MLAS_KV_QUANT_TYPE QuantType,
    const float* Scales,
    MLAS_THREADPOOL* ThreadPool)
{
    if (Rows == 0 || Cols == 0) {
        return;
    }

    const bool int4 = IsInt4Mode(QuantType);
    const bool per_channel = IsPerChannelMode(QuantType);
    const size_t row_bytes = MlasKVQuantPackedRowBytes(QuantType, Cols);
    auto* dst_bytes = static_cast<uint8_t*>(Dst);

    MlasTrySimpleParallel(
        ThreadPool, static_cast<ptrdiff_t>(Rows),
        [&](ptrdiff_t r_idx) {
            const size_t r = static_cast<size_t>(r_idx);
            const float* src_row = Src + r * lda;
            uint8_t* dst_row = dst_bytes + r * row_bytes;
            if (IsFp8Mode(QuantType)) {
                QuantizeRowFp8(src_row, dst_row, Cols, QuantType, Scales);
            } else if (int4) {
                QuantizeRowInt4(src_row, dst_row, Cols, per_channel, Scales);
            } else {
                QuantizeRowInt8(src_row, reinterpret_cast<int8_t*>(dst_row),
                                Cols, per_channel, Scales);
            }
        });
}

void
MLASCALL
MlasKVQuantizeFp16(
    const MLAS_FP16* Src,
    void* Dst,
    size_t Rows,
    size_t Cols,
    size_t lda,
    MLAS_KV_QUANT_TYPE QuantType,
    const float* Scales,
    MLAS_THREADPOOL* ThreadPool)
{
    if (Rows == 0 || Cols == 0) {
        return;
    }

    const bool int4 = IsInt4Mode(QuantType);
    const bool per_channel = IsPerChannelMode(QuantType);
    const size_t row_bytes = MlasKVQuantPackedRowBytes(QuantType, Cols);
    auto* dst_bytes = static_cast<uint8_t*>(Dst);

    MlasTrySimpleParallel(
        ThreadPool, static_cast<ptrdiff_t>(Rows),
        [&](ptrdiff_t r_idx) {
            const size_t r = static_cast<size_t>(r_idx);
            const MLAS_FP16* src_row = Src + r * lda;
            uint8_t* dst_row = dst_bytes + r * row_bytes;
            if (IsFp8Mode(QuantType)) {
                QuantizeRowFp8(src_row, dst_row, Cols, QuantType, Scales);
            } else if (int4) {
                QuantizeRowInt4Fp16(src_row, dst_row, Cols, per_channel, Scales);
            } else {
                QuantizeRowInt8Fp16(src_row, reinterpret_cast<int8_t*>(dst_row),
                                    Cols, per_channel, Scales);
            }
        });
}

void
MLASCALL
MlasKVDequantize(
    const void* Src,
    float* Dst,
    size_t Rows,
    size_t Cols,
    size_t ldb,
    MLAS_KV_QUANT_TYPE QuantType,
    const float* Scales,
    MLAS_THREADPOOL* ThreadPool)
{
    if (Rows == 0 || Cols == 0) {
        return;
    }

    const bool int4 = IsInt4Mode(QuantType);
    const bool per_channel = IsPerChannelMode(QuantType);
    const size_t row_bytes = MlasKVQuantPackedRowBytes(QuantType, Cols);
    const auto* src_bytes = static_cast<const uint8_t*>(Src);

    MlasTrySimpleParallel(
        ThreadPool, static_cast<ptrdiff_t>(Rows),
        [&](ptrdiff_t r_idx) {
            const size_t r = static_cast<size_t>(r_idx);
            const uint8_t* src_row = src_bytes + r * row_bytes;
            float* dst_row = Dst + r * ldb;
            if (IsFp8Mode(QuantType)) {
                DequantizeRowFp8(src_row, dst_row, Cols, QuantType, Scales);
            } else if (int4) {
                DequantizeRowInt4(src_row, dst_row, Cols, per_channel, Scales);
            } else {
                DequantizeRowInt8(reinterpret_cast<const int8_t*>(src_row),
                                  dst_row, Cols, per_channel, Scales);
            }
        });
}

void
MLASCALL
MlasQKGemm(
    size_t M,
    size_t N,
    size_t K,
    float Alpha,
    const float* A,
    size_t lda,
    const void* B,
    MLAS_KV_QUANT_TYPE QuantType,
    const float* Scales,
    float* C,
    size_t ldc,
    MLAS_THREADPOOL* ThreadPool)
{
    if (M == 0 || N == 0) {
        return;
    }
    if (K == 0) {
        for (size_t m = 0; m < M; ++m) {
            std::memset(C + m * ldc, 0, N * sizeof(float));
        }
        return;
    }

    if (UsesNativeInt8Gemm(QuantType)) {
        NativeInt8Workspace workspace;
        PrepareNativeInt8Query(M, K, A, lda, Scales, workspace);
        NativeInt8QKGemm(M, N, K, Alpha, B, C, ldc, workspace, ThreadPool);
        return;
    }

    if (UsesDecodedGemm(QuantType)) {
        std::vector<float> decoded(SafeInt<size_t>(N) * K);
        MlasKVDequantize(B, decoded.data(), N, K, K, QuantType, Scales, nullptr);
        MlasGemm(CblasNoTrans, CblasTrans, M, N, K, Alpha, A, lda, decoded.data(), K, 0.0f, C, ldc, ThreadPool, nullptr);
        return;
    }

    //
    // Try the SIMD-optimized dispatch path. The vectorized kernels handle
    // the full M×N×K computation in a single call (no thread pool — the
    // caller's thread-pool loop already partitions across heads/batches).
    //
    const auto* Dispatch = GetMlasPlatform().KVQuantGemmDispatch;
    if (Dispatch != nullptr && Dispatch->QKGemm != nullptr) {
        // The dispatch kernels are designed to be called per-(batch,head) tile
        // from an outer parallel loop, so we invoke them directly here. For
        // large N the outer loop in gqa_attention_base already parallelizes.
        Dispatch->QKGemm(M, N, K, Alpha, A, lda, B, QuantType, Scales, C, ldc);
        return;
    }

    //
    // Scalar reference fallback.
    //
    const bool int4 = IsInt4Mode(QuantType);
    const bool per_channel = IsPerChannelMode(QuantType);
    const size_t row_bytes = MlasKVQuantPackedRowBytes(QuantType, K);
    const auto* B_bytes = static_cast<const uint8_t*>(B);

    // Parallelize over N (independent output columns -> independent B rows).
    MlasTrySimpleParallel(
        ThreadPool, static_cast<ptrdiff_t>(N),
        [&](ptrdiff_t n_idx) {
            const size_t n = static_cast<size_t>(n_idx);
            const uint8_t* b_row = B_bytes + n * row_bytes;

            // Dequantize B row [K] once and reuse across all M.
            // Small buffer (K = head_size, typically <= 256).
            float b_dequant[1024];
            float* b_buf = b_dequant;
            std::unique_ptr<float[]> heap_buf;
            if (K > sizeof(b_dequant) / sizeof(b_dequant[0])) {
                heap_buf.reset(new float[K]);
                b_buf = heap_buf.get();
            }
            if (int4) {
                DequantizeRowInt4(b_row, b_buf, K, per_channel, Scales);
            } else {
                DequantizeRowInt8(reinterpret_cast<const int8_t*>(b_row),
                                  b_buf, K, per_channel, Scales);
            }

            for (size_t m = 0; m < M; ++m) {
                const float* a_row = A + m * lda;
                float acc = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    acc += a_row[k] * b_buf[k];
                }
                C[m * ldc + n] = Alpha * acc;
            }
        });
}

void
MLASCALL
MlasQKGemmFp16(
    size_t M,
    size_t N,
    size_t K,
    float Alpha,
    const MLAS_FP16* A,
    size_t lda,
    const void* B,
    MLAS_KV_QUANT_TYPE QuantType,
    const float* Scales,
    float* C,
    size_t ldc,
    MLAS_THREADPOOL* ThreadPool)
{
    if (M == 0 || N == 0) {
        return;
    }
    if (K == 0) {
        for (size_t m = 0; m < M; ++m) {
            std::memset(C + m * ldc, 0, N * sizeof(float));
        }
        return;
    }

    if (UsesNativeInt8Gemm(QuantType)) {
        NativeInt8Workspace workspace;
        PrepareNativeInt8Query(M, K, A, lda, Scales, workspace);
        NativeInt8QKGemm(M, N, K, Alpha, B, C, ldc, workspace, ThreadPool);
        return;
    }

    if (UsesDecodedGemm(QuantType)) {
        std::vector<float> query(SafeInt<size_t>(M) * K);
        for (size_t m = 0; m < M; ++m) {
            for (size_t k = 0; k < K; ++k) {
                query[m * K + k] = static_cast<float>(A[m * lda + k]);
            }
        }
        MlasQKGemm(M, N, K, Alpha, query.data(), K, B, QuantType, Scales, C, ldc, ThreadPool);
        return;
    }

    const auto* Dispatch = GetMlasPlatform().KVQuantGemmDispatch;
    if (GetMlasPlatform().KVQuantGemmFp16Supported_ &&
        Dispatch != nullptr && Dispatch->QKGemmFp16 != nullptr) {
        Dispatch->QKGemmFp16(M, N, K, Alpha, A, lda, B, QuantType, Scales, C, ldc);
        return;
    }

    const bool int4 = IsInt4Mode(QuantType);
    const bool per_channel = IsPerChannelMode(QuantType);
    const size_t row_bytes = MlasKVQuantPackedRowBytes(QuantType, K);
    const auto* B_bytes = static_cast<const uint8_t*>(B);

    MlasTrySimpleParallel(
        ThreadPool, static_cast<ptrdiff_t>(N),
        [&](ptrdiff_t n_idx) {
            const size_t n = static_cast<size_t>(n_idx);
            const uint8_t* b_row = B_bytes + n * row_bytes;
            for (size_t m = 0; m < M; ++m) {
                const MLAS_FP16* a_row = A + m * lda;
                float acc = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    const float b = int4
                                        ? DequantInt4FromByte(b_row[k / 2], k, per_channel ? Scales[k] : Scales[0])
                                        : DequantInt8(reinterpret_cast<const int8_t*>(b_row)[k], per_channel ? Scales[k] : Scales[0]);
                    acc += static_cast<float>(a_row[k]) * b;
                }
                C[m * ldc + n] = Alpha * acc;
            }
        });
}

void
MLASCALL
MlasSVGemm(
    size_t M,
    size_t N,
    size_t K,
    const float* A,
    size_t lda,
    const void* B,
    MLAS_KV_QUANT_TYPE QuantType,
    const float* Scales,
    float* C,
    size_t ldc,
    float Beta,
    MLAS_THREADPOOL* ThreadPool)
{
    if (M == 0 || N == 0) {
        return;
    }
    if (K == 0) {
        if (Beta == 0.0f) {
            for (size_t m = 0; m < M; ++m) {
                std::memset(C + m * ldc, 0, N * sizeof(float));
            }
        } else if (Beta != 1.0f) {
            for (size_t m = 0; m < M; ++m) {
                for (size_t n = 0; n < N; ++n) {
                    C[m * ldc + n] *= Beta;
                }
            }
        }
        return;
    }

    if (UsesNativeInt8Gemm(QuantType)) {
        NativeInt8Workspace workspace;
        NativeInt8SVGemm(M, N, K, A, lda, B, Scales, C, ldc, Beta, workspace, ThreadPool);
        return;
    }

    if (UsesDecodedGemm(QuantType)) {
        std::vector<float> decoded(SafeInt<size_t>(K) * N);
        MlasKVDequantize(B, decoded.data(), K, N, N, QuantType, Scales, nullptr);
        MlasGemm(CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, lda, decoded.data(), N, Beta, C, ldc, ThreadPool, nullptr);
        return;
    }

    //
    // Try the SIMD-optimized dispatch path.
    //
    const auto* Dispatch = GetMlasPlatform().KVQuantGemmDispatch;
    if (Dispatch != nullptr && Dispatch->SVGemm != nullptr) {
        Dispatch->SVGemm(M, N, K, A, lda, B, QuantType, Scales, C, ldc, Beta);
        return;
    }

    //
    // Scalar reference fallback.
    //
    const bool int4 = IsInt4Mode(QuantType);
    const bool per_channel = IsPerChannelMode(QuantType);
    const size_t row_bytes = MlasKVQuantPackedRowBytes(QuantType, N);
    const auto* B_bytes = static_cast<const uint8_t*>(B);

    // Parallelize over M (output rows). Each M iterates over the full K
    // reduction, dequantizing B rows one at a time.
    MlasTrySimpleParallel(
        ThreadPool, static_cast<ptrdiff_t>(M),
        [&](ptrdiff_t m_idx) {
            const size_t m = static_cast<size_t>(m_idx);
            const float* a_row = A + m * lda;
            float* c_row = C + m * ldc;
            if (Beta == 0.0f) {
                std::memset(c_row, 0, N * sizeof(float));
            } else if (Beta != 1.0f) {
                for (size_t n = 0; n < N; ++n) {
                    c_row[n] *= Beta;
                }
            }

            // Per-row scratch for one dequantized B row of length N.
            float b_dequant[1024];
            float* b_buf = b_dequant;
            std::unique_ptr<float[]> heap_buf;
            if (N > sizeof(b_dequant) / sizeof(b_dequant[0])) {
                heap_buf.reset(new float[N]);
                b_buf = heap_buf.get();
            }

            for (size_t k = 0; k < K; ++k) {
                const uint8_t* b_row_packed = B_bytes + k * row_bytes;
                if (int4) {
                    DequantizeRowInt4(b_row_packed, b_buf, N, per_channel, Scales);
                } else {
                    DequantizeRowInt8(reinterpret_cast<const int8_t*>(b_row_packed),
                                      b_buf, N, per_channel, Scales);
                }
                const float a_val = a_row[k];
                for (size_t n = 0; n < N; ++n) {
                    c_row[n] += a_val * b_buf[n];
                }
            }
        });
}


void
MLASCALL
MlasSVGemmFp16(
    size_t M,
    size_t N,
    size_t K,
    const float* A,
    size_t lda,
    const void* B,
    MLAS_KV_QUANT_TYPE QuantType,
    const float* Scales,
    MLAS_FP16* C,
    size_t ldc,
    float Beta,
    MLAS_THREADPOOL* ThreadPool)
{
    if (M == 0 || N == 0) {
        return;
    }
    if (K == 0) {
        for (size_t m = 0; m < M; ++m) {
            for (size_t n = 0; n < N; ++n) {
                C[m * ldc + n] = MLAS_FP16(Beta == 0.0f ? 0.0f : Beta * static_cast<float>(C[m * ldc + n]));
            }
        }
        return;
    }

    if (UsesDecodedGemm(QuantType) || UsesNativeInt8Gemm(QuantType)) {
        std::vector<float> output(SafeInt<size_t>(M) * N);
        if (Beta != 0.0f) {
            for (size_t m = 0; m < M; ++m) {
                for (size_t n = 0; n < N; ++n) {
                    output[m * N + n] = static_cast<float>(C[m * ldc + n]);
                }
            }
        }
        MlasSVGemm(M, N, K, A, lda, B, QuantType, Scales, output.data(), N, Beta, ThreadPool);
        for (size_t m = 0; m < M; ++m) {
            for (size_t n = 0; n < N; ++n) {
                C[m * ldc + n] = MLAS_FP16(output[m * N + n]);
            }
        }
        return;
    }

    const auto* Dispatch = GetMlasPlatform().KVQuantGemmDispatch;
    if (GetMlasPlatform().KVQuantGemmFp16Supported_ &&
        Dispatch != nullptr && Dispatch->SVGemmFp16 != nullptr) {
        Dispatch->SVGemmFp16(M, N, K, A, lda, B, QuantType, Scales, C, ldc, Beta);
        return;
    }

    const bool int4 = IsInt4Mode(QuantType);
    const bool per_channel = IsPerChannelMode(QuantType);
    const size_t row_bytes = MlasKVQuantPackedRowBytes(QuantType, N);
    const auto* B_bytes = static_cast<const uint8_t*>(B);

    MlasTrySimpleParallel(
        ThreadPool, static_cast<ptrdiff_t>(M),
        [&](ptrdiff_t m_idx) {
            const size_t m = static_cast<size_t>(m_idx);
            const float* a_row = A + m * lda;
            MLAS_FP16* c_row = C + m * ldc;
            for (size_t n = 0; n < N; ++n) {
                float acc = Beta == 0.0f ? 0.0f : Beta * static_cast<float>(c_row[n]);
                for (size_t k = 0; k < K; ++k) {
                    const uint8_t* b_row = B_bytes + k * row_bytes;
                    const float b = int4
                                        ? DequantInt4FromByte(b_row[n / 2], n, per_channel ? Scales[n] : Scales[0])
                                        : DequantInt8(reinterpret_cast<const int8_t*>(b_row)[n], per_channel ? Scales[n] : Scales[0]);
                    acc += a_row[k] * b;
                }
                c_row[n] = MLAS_FP16(acc);
            }
        });
}
