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
#include "mlas_qnbit.h"

#include <memory>
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

#if defined(MLAS_TARGET_ARM64) && !defined(DISABLE_FLOAT8_TYPES)
template <bool E5M2>
uint32x4_t QuantizeFp8x4(float32x4_t values)
{
    constexpr int shift = E5M2 ? 21 : 20;
    constexpr uint32_t minimum_normal = E5M2 ? 0x38800000U : 0x3c800000U;
    constexpr uint32_t bias = E5M2 ? 112U * 4 : 120U * 8;
    const uint32x4_t bits = vreinterpretq_u32_f32(values);
    const uint32x4_t magnitude = vandq_u32(bits, vdupq_n_u32(0x7fffffff));
    const uint32x4_t odd = vandq_u32(vshrq_n_u32(magnitude, shift), vdupq_n_u32(1));
    const uint32x4_t rounded = vaddq_u32(magnitude, vaddq_u32(vdupq_n_u32((1U << (shift - 1)) - 1), odd));
    const uint32x4_t normal = vminq_u32(
        vsubq_u32(vshrq_n_u32(rounded, shift), vdupq_n_u32(bias)), vdupq_n_u32(E5M2 ? 0x7b : 0x7e));
    const uint32x4_t small = vcltq_u32(magnitude, vdupq_n_u32(minimum_normal));
    // Mask before conversion so NaN/Inf and large normal values cannot overflow the subnormal calculation.
    const float32x4_t subnormal = vreinterpretq_f32_u32(vandq_u32(magnitude, small));
    const uint32x4_t subnormal_bits = vcvtnq_u32_f32(vmulq_n_f32(subnormal, E5M2 ? 65536.0f : 512.0f));
    uint32x4_t encoded = vbslq_u32(small, subnormal_bits, normal);
    encoded = vbslq_u32(vcgtq_u32(magnitude, vdupq_n_u32(0x7f800000)), vdupq_n_u32(0x7f), encoded);
    return vorrq_u32(encoded, vshrq_n_u32(vandq_u32(bits, vdupq_n_u32(0x80000000)), 24));
}

float32x4_t Fp8InverseScales(const float* scales)
{
    const float32x4_t values = vld1q_f32(scales);
    const float32x4_t nonzero = vbslq_f32(vceqq_f32(values, vdupq_n_f32(0.0f)), vdupq_n_f32(1.0f), values);
    return vdivq_f32(vdupq_n_f32(1.0f), nonzero);
}

float16x8_t DecodeFp8x8(const uint8_t* source, bool e4m3)
{
    const uint8x8_t bytes = vld1_u8(source);
    const uint16x8_t magnitude = vmovl_u8(vand_u8(bytes, vdup_n_u8(0x7f)));
    uint16x8_t bits;
    if (e4m3) {
        static constexpr uint8_t subnormal_high_bytes[8] = {0, 0x18, 0x1c, 0x1e, 0x20, 0x21, 0x22, 0x23};
        const uint8x8_t small = vtbl1_u8(vld1_u8(subnormal_high_bytes), vand_u8(bytes, vdup_n_u8(7)));
        bits = vbslq_u16(
            vcltq_u16(magnitude, vdupq_n_u16(8)), vshll_n_u8(small, 8),
            vaddq_u16(vshlq_n_u16(magnitude, 7), vdupq_n_u16(0x2000)));
        bits = vbslq_u16(vceqq_u16(magnitude, vdupq_n_u16(0x7f)), vdupq_n_u16(0x7e00), bits);
    } else {
        bits = vshlq_n_u16(magnitude, 8);
        bits = vbslq_u16(vcgtq_u16(magnitude, vdupq_n_u16(0x7c)), vdupq_n_u16(0x7e00), bits);
    }
    return vreinterpretq_f16_u16(vorrq_u16(bits, vshll_n_u8(vand_u8(bytes, vdup_n_u8(0x80)), 8)));
}
#endif

template <typename T>
void
QuantizeRowFp8(const T* src, uint8_t* dst, size_t cols, MLAS_KV_QUANT_TYPE type, const float* scales)
{
#if !defined(DISABLE_FLOAT8_TYPES)
    const bool e4m3 = IsFp8E4M3Mode(type);
    size_t c = 0;
#if defined(MLAS_TARGET_ARM64)
    for (; c + 8 <= cols; c += 8) {
        float32x4_t low, high;
        if constexpr (std::is_same_v<T, float>) {
            low = vld1q_f32(src + c);
            high = vld1q_f32(src + c + 4);
        } else {
            const float16x8_t half = vreinterpretq_f16_u16(vld1q_u16(reinterpret_cast<const uint16_t*>(src + c)));
            low = vcvt_f32_f16(vget_low_f16(half));
            high = vcvt_f32_f16(vget_high_f16(half));
        }
        low = vmulq_f32(low, Fp8InverseScales(scales + c));
        high = vmulq_f32(high, Fp8InverseScales(scales + c + 4));
        const uint32x4_t low_bits = e4m3 ? QuantizeFp8x4<false>(low) : QuantizeFp8x4<true>(low);
        const uint32x4_t high_bits = e4m3 ? QuantizeFp8x4<false>(high) : QuantizeFp8x4<true>(high);
        vst1_u8(dst + c, vmovn_u16(vcombine_u16(vmovn_u32(low_bits), vmovn_u32(high_bits))));
    }
#endif
    for (; c < cols; ++c) {
        const float value = static_cast<float>(src[c]) * SafeInvScale(scales[c]);
        dst[c] = e4m3
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
    return IsFp8E4M3Mode(type) ? e4m3 : e5m2;
#else
    ORT_UNUSED_PARAMETER(type);
    ORT_THROW("FP8 KV support is disabled in this build");
#endif
}

template <typename T>
void
DequantizeRowFp8(const uint8_t* src, T* dst, size_t cols, MLAS_KV_QUANT_TYPE type, const float* scales)
{
    size_t c = 0;
#if defined(MLAS_TARGET_ARM64) && !defined(DISABLE_FLOAT8_TYPES)
    for (; c + 8 <= cols; c += 8) {
        const float16x8_t half = DecodeFp8x8(src + c, IsFp8E4M3Mode(type));
        const float32x4_t low = vmulq_f32(vcvt_f32_f16(vget_low_f16(half)), vld1q_f32(scales + c));
        const float32x4_t high = vmulq_f32(vcvt_f32_f16(vget_high_f16(half)), vld1q_f32(scales + c + 4));
        if constexpr (std::is_same_v<T, float>) {
            vst1q_f32(dst + c, low);
            vst1q_f32(dst + c + 4, high);
        } else {
            const float16x8_t scaled = vcombine_f16(vcvt_f16_f32(low), vcvt_f16_f32(high));
            vst1q_u16(reinterpret_cast<uint16_t*>(dst + c), vreinterpretq_u16_f16(scaled));
        }
    }
    if (c == cols) {
        return;
    }
#endif
    const auto& values = Fp8Lookup(type);
    for (; c < cols; ++c) {
        dst[c] = T(values[src[c]] * scales[c]);
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

void ValidateKVScales(const float* scales, size_t count, const char* name)
{
    ORT_ENFORCE(scales != nullptr, name, " scales are required");
    for (size_t i = 0; i < count; ++i) {
        ORT_ENFORCE(std::isfinite(scales[i]) && scales[i] > 0.0f,
                    name, " scales must be finite and positive");
    }
}

void ValidateNativeInt8Scales(const float* scales, size_t count)
{
    ORT_ENFORCE(MlasIsKVQuantGemmSupported(MLAS_KV_QUANT_TYPE::S8_PerChannel_Native),
                "Native INT8 KV attention requires ARM64 integer GEMM support");
    ValidateKVScales(scales, count, "Native INT8 KV");
}

float NativeInt8Scale(float peak, float limit)
{
    // Tiny online-softmax tiles must not underflow the quantization scale to zero.
    return peak == 0.0f ? 1.0f : std::max(peak / limit, std::numeric_limits<float>::min());
}

bool FindFiniteRange(const float* input, size_t count, float& minimum, float& maximum)
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
void PrepareScaledQueryRow(const T* input, size_t count, const float* scales, float* row)
{
    if constexpr (std::is_same_v<T, float>) {
        std::copy_n(input, count, row);
    } else {
        MlasConvertHalfToFloatBuffer(input, row, count);
    }
    size_t k = 0;
#if defined(MLAS_TARGET_ARM64)
    for (; k + 4 <= count; k += 4) {
        vst1q_f32(row + k, vmulq_f32(vld1q_f32(row + k), vld1q_f32(scales + k)));
    }
#endif
    for (; k < count; ++k) {
        row[k] *= scales[k];
    }
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
        // Fold per-channel K scaling into Q before its per-row quantization.
        PrepareScaledQueryRow(A + m * lda, K, scales, row);
        float minimum, maximum;
        ORT_ENFORCE(FindFiniteRange(row, K, minimum, maximum),
                    "Native INT8 attention requires finite scaled queries");
        row_scales[m] = NativeInt8Scale(std::max(std::abs(minimum), std::abs(maximum)), 127.0f);
        MlasQuantizeLinear(row, query + m * K, K, row_scales[m], int8_t{0});
    }
    workspace.query_rows = M;
    workspace.query_columns = K;
}

constexpr size_t NativeInt4BlockSize = 128;

size_t NativeInt4Blocks(size_t count)
{
    return count / NativeInt4BlockSize + (count % NativeInt4BlockSize != 0);
}

bool NativeInt4UseInt8(size_t reduction)
{
    return MlasIsKVQuantGemmSupported(MLAS_KV_QUANT_TYPE::S8_PerChannel_Native) &&
           reduction <= static_cast<size_t>(std::numeric_limits<int32_t>::max()) / (255 * 128);
}

void UnpackNativeInt4Rows(size_t rows, size_t columns, const void* source, int8_t* target)
{
    const size_t stride = columns / 2 + columns % 2;
    for (size_t row = 0; row < rows; ++row) {
        const auto* input = static_cast<const uint8_t*>(source) + row * stride;
        auto* output = target + row * columns;
        size_t col = 0;
#if defined(MLAS_TARGET_ARM64)
        for (; col + 16 <= columns; col += 16) {
            const uint8x8_t packed = vld1_u8(input + col / 2);
            const uint8x8x2_t pairs = vzip_u8(vand_u8(packed, vdup_n_u8(0x0f)), vshr_n_u8(packed, 4));
            const int8x16_t values = vreinterpretq_s8_u8(vcombine_u8(pairs.val[0], pairs.val[1]));
            vst1q_s8(output + col, vsubq_s8(values, vdupq_n_s8(8)));
        }
#endif
        for (; col < columns; ++col) {
            output[col] = static_cast<int8_t>(static_cast<int>((input[col / 2] >> ((col % 2) * 4)) & 0x0f) - 8);
        }
    }
}

void* AlignedWorkspaceBuffer(std::vector<uint8_t>& storage, size_t bytes)
{
    if (bytes == 0) {
        return nullptr;
    }
    const size_t alignment = MlasGetPreferredBufferAlignment();
    void* result = WorkspaceBuffer(storage, SafeInt<size_t>(bytes) + alignment - 1);
    size_t space = storage.size();
    ORT_ENFORCE(std::align(alignment, bytes, result, space) != nullptr, "Cannot align native INT4 scratch");
    return result;
}

template <typename T>
void PrepareNativeInt4QueryImpl(
    size_t M, size_t K, const T* A, size_t lda, const float* scales, NativeInt4Workspace& workspace)
{
    ORT_ENFORCE(MlasIsKVQuantGemmSupported(MLAS_KV_QUANT_TYPE::S4_PerChannel_Native),
                "Native INT4 KV attention requires integer GEMM support");
    ValidateKVScales(scales, K, "Native INT4 KV");
    ORT_ENFORCE(K > 0 && lda >= K, "Invalid native INT4 query dimensions or row stride");
    workspace.query_rows = workspace.query_columns = 0;
    if (NativeInt4UseInt8(K)) {
        // Dynamic KV repacking is costly for weight kernels; reuse ARM64 integer attention.
        PrepareNativeInt8QueryImpl(M, K, A, lda, scales, workspace.native_int8);
        workspace.query_rows = M;
        workspace.query_columns = K;
        return;
    }
    auto* query = WorkspaceBuffer(workspace.query, SafeInt<size_t>(M) * K);
    for (size_t m = 0; m < M; ++m) {
        auto* row = query + m * K;
        PrepareScaledQueryRow(A + m * lda, K, scales, row);
        float minimum, maximum;
        ORT_ENFORCE(FindFiniteRange(row, K, minimum, maximum),
                    "Native INT4 attention requires finite scaled queries");
    }
    workspace.query_rows = M;
    workspace.query_columns = K;
}

void PackNativeInt4Values(size_t K, size_t N, const uint8_t* source, uint8_t* target, size_t stride)
{
    const size_t source_stride = N / 2 + N % 2;
    if (K % NativeInt4BlockSize != 0) {
        const size_t bytes = SafeInt<size_t>(N) * stride;
        std::fill_n(target, bytes, uint8_t{0x88});
    }
    const size_t pairs = K / 2;
    size_t pair = 0;
#if defined(MLAS_TARGET_ARM64)
    const uint8x8_t low_mask = vdup_n_u8(0x0f);
    const uint8x8_t high_mask = vdup_n_u8(0xf0);
    alignas(16) uint8_t tile[8 * 16], transposed[8 * 16];
    for (; pair + 8 <= pairs; pair += 8) {
        size_t col = 0;
        for (; col + 16 <= N; col += 16) {
            for (size_t row = 0; row < 8; ++row) {
                const auto* first = source + 2 * (pair + row) * source_stride + col / 2;
                const uint8x8_t a = vld1_u8(first);
                const uint8x8_t b = vld1_u8(first + source_stride);
                const uint8x8_t even = vorr_u8(vand_u8(a, low_mask), vshl_n_u8(b, 4));
                const uint8x8_t odd = vorr_u8(vshr_n_u8(a, 4), vand_u8(b, high_mask));
                const uint8x8x2_t columns = vzip_u8(even, odd);
                vst1q_u8(tile + row * 16, vcombine_u8(columns.val[0], columns.val[1]));
            }
            MlasTranspose(tile, transposed, 8, 16, nullptr);
            for (size_t i = 0; i < 16; ++i) {
                vst1_u8(target + (col + i) * stride + pair, vld1_u8(transposed + i * 8));
            }
        }
        for (size_t row = 0; row < 8; ++row) {
            for (size_t i = col; i < N; ++i) {
                const size_t shift = (i % 2) * 4;
                const auto a = source[2 * (pair + row) * source_stride + i / 2];
                const auto b = source[(2 * (pair + row) + 1) * source_stride + i / 2];
                target[i * stride + pair + row] = static_cast<uint8_t>(
                    ((a >> shift) & 0x0f) | (((b >> shift) & 0x0f) << 4));
            }
        }
    }
#endif
    for (; pair < pairs; ++pair) {
        for (size_t col = 0; col < N; ++col) {
            const size_t shift = (col % 2) * 4;
            const auto a = source[2 * pair * source_stride + col / 2];
            const auto b = source[(2 * pair + 1) * source_stride + col / 2];
            target[col * stride + pair] = static_cast<uint8_t>(
                ((a >> shift) & 0x0f) | (((b >> shift) & 0x0f) << 4));
        }
    }
    if (K % 2 != 0) {
        for (size_t col = 0; col < N; ++col) {
            const auto last = source[(K - 1) * source_stride + col / 2];
            target[col * stride + pairs] = static_cast<uint8_t>(((last >> ((col % 2) * 4)) & 0x0f) | 0x80);
        }
    }
}

void NativeInt4Product(
    size_t M, size_t N, size_t K, const float* A, size_t lda, const void* B, const float* scales,
    float* C, size_t ldc, NativeInt4Workspace& workspace, MLAS_THREADPOOL* thread_pool)
{
    constexpr auto compute = SQNBIT_CompInt8;
    ORT_ENFORCE(MlasIsQNBitGemmAvailable(4, NativeInt4BlockSize, compute),
                "This native INT4 reduction requires blocked integer GEMM support");
    const size_t pack_size = MlasQNBitGemmPackQuantBDataSize(N, K, 4, NativeInt4BlockSize, false, compute, nullptr);
    const void* packed = B;
    if (pack_size != 0) {
        auto* buffer = AlignedWorkspaceBuffer(workspace.packed, pack_size);
        MlasQNBitGemmPackQuantBData(N, K, 4, NativeInt4BlockSize, compute, B, buffer,
                                  scales, false, nullptr, thread_pool, nullptr);
        bool finalize = MlasQNBitGemmScalesPacked(K, 4, NativeInt4BlockSize, compute, false, nullptr);
#if defined(MLAS_TARGET_AMD64_IX86)
        finalize = true;
#endif
        if (finalize) {
            MlasQNBitGemmPackQuantBData(N, K, 4, NativeInt4BlockSize, compute, nullptr, buffer,
                                      scales, false, nullptr, thread_pool, nullptr);
        }
        packed = buffer;
    }
    const size_t workspace_size =
        MlasQNBitGemmBatchWorkspaceSize(M, N, K, 1, 4, NativeInt4BlockSize, false, compute, nullptr);
    auto* scratch = AlignedWorkspaceBuffer(workspace.gemm_workspace, workspace_size);
    MLAS_QNBIT_GEMM_DATA_PARAMS<float> data{};
    data.A = A;
    data.lda = lda;
    data.QuantBDataWorkspace = packed;
    data.PackedQuantBData = static_cast<const std::byte*>(packed);
    data.QuantBScale = scales;
    data.C = C;
    data.ldc = ldc;
    MlasQNBitGemmBatch(M, N, K, 1, 4, NativeInt4BlockSize, compute, &data, scratch, thread_pool, nullptr);
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

void PrepareNativeInt4Query(
    size_t M, size_t K, const float* A, size_t lda, const float* Scales, NativeInt4Workspace& Workspace)
{
    PrepareNativeInt4QueryImpl(M, K, A, lda, Scales, Workspace);
}

void PrepareNativeInt4Query(
    size_t M, size_t K, const MLAS_FP16* A, size_t lda, const float* Scales, NativeInt4Workspace& Workspace)
{
    PrepareNativeInt4QueryImpl(M, K, A, lda, Scales, Workspace);
}

void NativeInt4QKGemm(
    size_t M, size_t N, size_t K, float Alpha, const void* B, float* C, size_t ldc,
    NativeInt4Workspace& Workspace, MLAS_THREADPOOL* ThreadPool)
{
    ORT_ENFORCE(Workspace.query_rows == M && Workspace.query_columns == K && ldc >= N,
                "Native INT4 query tile or output stride is invalid");
    ORT_ENFORCE(std::isfinite(Alpha), "Native INT4 attention requires a finite QK scale");
    if (Alpha == 0.0f || K == 0) {
        for (size_t m = 0; m < M; ++m) std::fill_n(C + m * ldc, N, 0.0f);
        return;
    }
    if (NativeInt4UseInt8(K)) {
        auto* expanded = WorkspaceBuffer(Workspace.unpacked, SafeInt<size_t>(N) * K);
        UnpackNativeInt4Rows(N, K, B, expanded);
        NativeInt8QKGemm(M, N, K, Alpha, expanded, C, ldc, Workspace.native_int8, ThreadPool);
        return;
    }
    const size_t blocks = NativeInt4Blocks(K);
    const size_t stride = SafeInt<size_t>(blocks) * (NativeInt4BlockSize / 2);
    const void* blocked = B;
    if (K % NativeInt4BlockSize != 0) {
        const size_t bytes = SafeInt<size_t>(N) * stride;
        auto* buffer = WorkspaceBuffer(Workspace.blocked, bytes);
        std::fill_n(buffer, bytes, uint8_t{0x88});
        const size_t row_bytes = K / 2 + K % 2;
        for (size_t n = 0; n < N; ++n) {
            std::memcpy(buffer + n * stride, static_cast<const uint8_t*>(B) + n * row_bytes, row_bytes);
            if (K % 2 != 0) {
                buffer[n * stride + row_bytes - 1] = static_cast<uint8_t>(
                    (buffer[n * stride + row_bytes - 1] & 0x0f) | 0x80);
            }
        }
        blocked = buffer;
    }
    const size_t scale_count = SafeInt<size_t>(N) * blocks;
    auto* scales = WorkspaceBuffer(Workspace.scales, scale_count);
    std::fill_n(scales, scale_count, Alpha);
    NativeInt4Product(M, N, K, Workspace.query.data(), K, blocked, scales, C, ldc, Workspace, ThreadPool);
}

void NativeInt4SVGemm(
    size_t M, size_t N, size_t K, const float* A, size_t lda, const void* B, const float* Scales,
    float* C, size_t ldc, float Beta, NativeInt4Workspace& Workspace, MLAS_THREADPOOL* ThreadPool)
{
    ORT_ENFORCE(MlasIsKVQuantGemmSupported(MLAS_KV_QUANT_TYPE::S4_PerChannel_Native),
                "Native INT4 KV attention requires integer GEMM support");
    ORT_ENFORCE(K > 0 && lda >= K && ldc >= N, "Invalid native INT4 probability or output row stride");
    ValidateKVScales(Scales, N, "Native INT4 KV");
    if (MlasIsKVQuantGemmSupported(MLAS_KV_QUANT_TYPE::S8_PerChannel_Native)) {
        auto* expanded = WorkspaceBuffer(Workspace.unpacked, SafeInt<size_t>(K) * N);
        UnpackNativeInt4Rows(K, N, B, expanded);
        // Short probability spans preserve weak attention beside a stronger local peak.
        for (size_t offset = 0; offset < K; offset += 128) {
            NativeInt8SVGemm(M, N, std::min(size_t{128}, K - offset), A + offset, lda,
                            expanded + offset * N, Scales, C, ldc, offset == 0 ? Beta : 1.0f,
                            Workspace.native_int8, ThreadPool);
        }
        return;
    }
    for (size_t m = 0; m < M; ++m) {
        float minimum, maximum;
        ORT_ENFORCE(FindFiniteRange(A + m * lda, K, minimum, maximum),
                    "Native INT4 attention requires finite probabilities");
    }
    const size_t blocks = NativeInt4Blocks(K);
    const size_t stride = SafeInt<size_t>(blocks) * (NativeInt4BlockSize / 2);
    auto* blocked = WorkspaceBuffer(Workspace.blocked, SafeInt<size_t>(N) * stride);
    PackNativeInt4Values(K, N, static_cast<const uint8_t*>(B), blocked, stride);
    auto* scales = WorkspaceBuffer(Workspace.scales, SafeInt<size_t>(N) * blocks);
    for (size_t n = 0; n < N; ++n) std::fill_n(scales + n * blocks, blocks, Scales[n]);
    float* products = Beta == 0.0f ? C : WorkspaceBuffer(Workspace.products, SafeInt<size_t>(M) * N);
    NativeInt4Product(M, N, K, A, lda, blocked, scales, products, Beta == 0.0f ? ldc : N, Workspace, ThreadPool);
    if (Beta == 0.0f) {
        return;
    }
    for (size_t m = 0; m < M; ++m) {
        size_t n = 0;
#if defined(MLAS_TARGET_ARM64)
        for (; n + 4 <= N; n += 4) {
            vst1q_f32(C + m * ldc + n, vaddq_f32(vld1q_f32(products + m * N + n),
                                                vmulq_n_f32(vld1q_f32(C + m * ldc + n), Beta)));
        }
#endif
        for (; n < N; ++n) C[m * ldc + n] = products[m * N + n] + Beta * C[m * ldc + n];
    }
}

void PrepareDecodedQuery(
    size_t M, size_t K, const float* A, size_t lda, DecodedGemmWorkspace& Workspace)
{
    ORT_ENFORCE(lda >= K, "Invalid decoded KV query row stride");
    Workspace.prepared_query = A;
    Workspace.query_stride = lda;
    Workspace.query_rows = M;
    Workspace.query_columns = K;
}

void PrepareDecodedQuery(
    size_t M, size_t K, const MLAS_FP16* A, size_t lda, DecodedGemmWorkspace& Workspace)
{
    ORT_ENFORCE(lda >= K, "Invalid decoded KV query row stride");
    auto* query = WorkspaceBuffer(Workspace.query, SafeInt<size_t>(M) * K);
    for (size_t m = 0; m < M; ++m) {
        MlasConvertHalfToFloatBuffer(A + m * lda, query + m * K, K);
    }
    PrepareDecodedQuery(M, K, query, K, Workspace);
}

void DecodedQKGemm(
    size_t M, size_t N, size_t K, float Alpha, const void* B, MLAS_KV_QUANT_TYPE QuantType,
    const float* Scales, float* C, size_t ldc, DecodedGemmWorkspace& Workspace, MLAS_THREADPOOL* ThreadPool)
{
    ORT_ENFORCE(UsesDecodedGemm(QuantType), "Decoded KV GEMM requires a decoded-compute cache mode");
    ORT_ENFORCE(Workspace.query_rows == M && Workspace.query_columns == K && ldc >= N,
                "Decoded KV query tile or output stride is invalid");
    if (K == 0 || Alpha == 0.0f) {
        for (size_t m = 0; m < M; ++m) {
            std::fill_n(C + m * ldc, N, 0.0f);
        }
        return;
    }
    auto* decoded = WorkspaceBuffer(Workspace.decoded, SafeInt<size_t>(N) * K);
    MlasKVDequantize(B, decoded, N, K, K, QuantType, Scales, nullptr);
    MlasGemm(CblasNoTrans, CblasTrans, M, N, K, Alpha, Workspace.prepared_query, Workspace.query_stride,
             decoded, K, 0.0f, C, ldc, ThreadPool, nullptr);
}

void DecodedSVGemm(
    size_t M, size_t N, size_t K, const float* A, size_t lda, const void* B,
    MLAS_KV_QUANT_TYPE QuantType, const float* Scales, float* C, size_t ldc, float Beta,
    DecodedGemmWorkspace& Workspace, MLAS_THREADPOOL* ThreadPool)
{
    ORT_ENFORCE(UsesDecodedGemm(QuantType), "Decoded KV GEMM requires a decoded-compute cache mode");
    ORT_ENFORCE(lda >= K && ldc >= N, "Invalid decoded KV probability or output row stride");
    auto* decoded = WorkspaceBuffer(Workspace.decoded, SafeInt<size_t>(K) * N);
    MlasKVDequantize(B, decoded, K, N, N, QuantType, Scales, nullptr);
    MlasGemm(CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, lda, decoded, N, Beta, C, ldc, ThreadPool, nullptr);
}

void PrepareFp8HalfQuery(
    size_t M, size_t K, const float* A, size_t lda, Fp8HalfWorkspace& Workspace)
{
    ORT_ENFORCE(lda >= K, "Invalid FP8 query row stride");
    auto* query = WorkspaceBuffer(Workspace.query, SafeInt<size_t>(M) * K);
    for (size_t m = 0; m < M; ++m) {
        MlasConvertFloatToHalfBuffer(A + m * lda, query + m * K, K);
    }
    PrepareFp8HalfQuery(M, K, query, K, Workspace);
}

void PrepareFp8HalfQuery(
    size_t M, size_t K, const MLAS_FP16* A, size_t lda, Fp8HalfWorkspace& Workspace)
{
    ORT_ENFORCE(lda >= K, "Invalid FP8 query row stride");
    Workspace.prepared_query = A;
    Workspace.query_stride = lda;
    Workspace.query_rows = M;
    Workspace.query_columns = K;
}

void Fp8HalfQKGemm(
    size_t M, size_t N, size_t K, float Alpha, const void* B, MLAS_KV_QUANT_TYPE QuantType,
    const float* Scales, float* C, size_t ldc, Fp8HalfWorkspace& Workspace, MLAS_THREADPOOL* ThreadPool)
{
    ORT_ENFORCE(UsesFp8HalfGemm(QuantType) && MlasIsKVQuantGemmSupported(QuantType),
                "FP8 FP16 compute requires accelerated half GEMMs");
    ORT_ENFORCE(Workspace.query_rows == M && Workspace.query_columns == K && ldc >= N,
                "FP8 FP16 query tile or output stride is invalid");
    ValidateKVScales(Scales, K, "FP8 KV");
    auto* decoded = WorkspaceBuffer(Workspace.decoded, SafeInt<size_t>(N) * K);
    auto* products = WorkspaceBuffer(Workspace.products, SafeInt<size_t>(M) * N);
    for (size_t n = 0; n < N; ++n) {
        DequantizeRowFp8(static_cast<const uint8_t*>(B) + n * K, decoded + n * K, K, QuantType, Scales);
    }
    MlasGemm(CblasNoTrans, CblasTrans, M, N, K, Workspace.prepared_query, Workspace.query_stride,
             decoded, K, products, N, MLAS_FP16(Alpha).val, uint16_t{0}, ThreadPool);
    for (size_t m = 0; m < M; ++m) {
        MlasConvertHalfToFloatBuffer(products + m * N, C + m * ldc, N);
    }
}

void Fp8HalfSVGemm(
    size_t M, size_t N, size_t K, const float* A, size_t lda, const void* B,
    MLAS_KV_QUANT_TYPE QuantType, const float* Scales, float* C, size_t ldc, float Beta,
    Fp8HalfWorkspace& Workspace, MLAS_THREADPOOL* ThreadPool)
{
    ORT_ENFORCE(UsesFp8HalfGemm(QuantType) && MlasIsKVQuantGemmSupported(QuantType),
                "FP8 FP16 compute requires accelerated half GEMMs");
    ORT_ENFORCE(lda >= K && ldc >= N, "Invalid FP8 probability or output row stride");
    ValidateKVScales(Scales, N, "FP8 KV");
    if (K == 0) {
        for (size_t m = 0; m < M; ++m) {
            for (size_t n = 0; n < N; ++n) {
                C[m * ldc + n] = Beta == 0.0f ? 0.0f : Beta * C[m * ldc + n];
            }
        }
        return;
    }
    auto* decoded = WorkspaceBuffer(Workspace.decoded, SafeInt<size_t>(K) * N);
    auto* probabilities = WorkspaceBuffer(Workspace.probabilities, SafeInt<size_t>(M) * K);
    auto* products = WorkspaceBuffer(Workspace.products, SafeInt<size_t>(M) * N);
    for (size_t k = 0; k < K; ++k) {
        DequantizeRowFp8(static_cast<const uint8_t*>(B) + k * N, decoded + k * N, N, QuantType, Scales);
    }
    for (size_t m = 0; m < M; ++m) {
        MlasConvertFloatToHalfBuffer(A + m * lda, probabilities + m * K, K);
    }
    // Single-row HGEMM otherwise accumulates the entire KV tile in FP16.
    for (size_t offset = 0; offset < K; offset += 128) {
        const size_t count = std::min(size_t{128}, K - offset);
        const float beta = offset == 0 ? Beta : 1.0f;
        MlasGemm(CblasNoTrans, CblasNoTrans, M, N, count, probabilities + offset, K,
                 decoded + offset * N, N, products, N, MLAS_FP16(1.0f).val, uint16_t{0}, ThreadPool);
        for (size_t m = 0; m < M; ++m) {
            if (beta == 0.0f) {
                MlasConvertHalfToFloatBuffer(products + m * N, C + m * ldc, N);
                continue;
            }
            size_t n = 0;
#if defined(MLAS_TARGET_ARM64)
            for (; n + 8 <= N; n += 8) {
                const float16x8_t half = vreinterpretq_f16_u16(
                    vld1q_u16(reinterpret_cast<const uint16_t*>(products + m * N + n)));
                vst1q_f32(C + m * ldc + n,
                          vaddq_f32(vcvt_f32_f16(vget_low_f16(half)),
                                    vmulq_n_f32(vld1q_f32(C + m * ldc + n), beta)));
                vst1q_f32(C + m * ldc + n + 4,
                          vaddq_f32(vcvt_f32_f16(vget_high_f16(half)),
                                    vmulq_n_f32(vld1q_f32(C + m * ldc + n + 4), beta)));
            }
#endif
            for (; n < N; ++n) {
                C[m * ldc + n] = beta * C[m * ldc + n] + static_cast<float>(products[m * N + n]);
            }
        }
    }
}

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
        ORT_ENFORCE(FindFiniteRange(row, K, minimum, maximum),
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
    if (UsesNativeInt4Gemm(QuantType)) {
        return MlasIsKVQuantGemmSupported(MLAS_KV_QUANT_TYPE::S8_PerChannel_Native) ||
               MlasIsQNBitGemmAvailable(4, NativeInt4BlockSize, SQNBIT_CompInt8);
    }
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
    if (UsesFp8HalfGemm(QuantType)) {
        return MlasHGemmSupported(CblasNoTrans, CblasTrans) &&
               MlasHGemmSupported(CblasNoTrans, CblasNoTrans);
    }
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

    if (UsesNativeInt4Gemm(QuantType)) {
        NativeInt4Workspace workspace;
        PrepareNativeInt4Query(M, K, A, lda, Scales, workspace);
        NativeInt4QKGemm(M, N, K, Alpha, B, C, ldc, workspace, ThreadPool);
        return;
    }
    if (UsesNativeInt8Gemm(QuantType)) {
        NativeInt8Workspace workspace;
        PrepareNativeInt8Query(M, K, A, lda, Scales, workspace);
        NativeInt8QKGemm(M, N, K, Alpha, B, C, ldc, workspace, ThreadPool);
        return;
    }

    if (UsesFp8HalfGemm(QuantType)) {
        Fp8HalfWorkspace workspace;
        PrepareFp8HalfQuery(M, K, A, lda, workspace);
        Fp8HalfQKGemm(M, N, K, Alpha, B, QuantType, Scales, C, ldc, workspace, ThreadPool);
        return;
    }

    if (UsesDecodedGemm(QuantType)) {
        DecodedGemmWorkspace workspace;
        PrepareDecodedQuery(M, K, A, lda, workspace);
        DecodedQKGemm(M, N, K, Alpha, B, QuantType, Scales, C, ldc, workspace, ThreadPool);
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

    if (UsesNativeInt4Gemm(QuantType)) {
        NativeInt4Workspace workspace;
        PrepareNativeInt4Query(M, K, A, lda, Scales, workspace);
        NativeInt4QKGemm(M, N, K, Alpha, B, C, ldc, workspace, ThreadPool);
        return;
    }
    if (UsesNativeInt8Gemm(QuantType)) {
        NativeInt8Workspace workspace;
        PrepareNativeInt8Query(M, K, A, lda, Scales, workspace);
        NativeInt8QKGemm(M, N, K, Alpha, B, C, ldc, workspace, ThreadPool);
        return;
    }

    if (UsesFp8HalfGemm(QuantType)) {
        Fp8HalfWorkspace workspace;
        PrepareFp8HalfQuery(M, K, A, lda, workspace);
        Fp8HalfQKGemm(M, N, K, Alpha, B, QuantType, Scales, C, ldc, workspace, ThreadPool);
        return;
    }

    if (UsesDecodedGemm(QuantType)) {
        DecodedGemmWorkspace workspace;
        PrepareDecodedQuery(M, K, A, lda, workspace);
        DecodedQKGemm(M, N, K, Alpha, B, QuantType, Scales, C, ldc, workspace, ThreadPool);
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

    if (UsesNativeInt4Gemm(QuantType)) {
        NativeInt4Workspace workspace;
        NativeInt4SVGemm(M, N, K, A, lda, B, Scales, C, ldc, Beta, workspace, ThreadPool);
        return;
    }
    if (UsesNativeInt8Gemm(QuantType)) {
        NativeInt8Workspace workspace;
        NativeInt8SVGemm(M, N, K, A, lda, B, Scales, C, ldc, Beta, workspace, ThreadPool);
        return;
    }

    if (UsesFp8HalfGemm(QuantType)) {
        if (K > 128) {
            ORT_ENFORCE(MlasIsKVQuantGemmSupported(QuantType),
                        "FP8 FP16 compute requires accelerated half GEMMs");
            ValidateKVScales(Scales, N, "FP8 KV");
            const auto decoded_type = IsFp8E4M3Mode(QuantType)
                ? MLAS_KV_QUANT_TYPE::F8E4M3FN_PerChannel
                : MLAS_KV_QUANT_TYPE::F8E5M2_PerChannel;
            DecodedGemmWorkspace workspace;
            DecodedSVGemm(M, N, K, A, lda, B, decoded_type, Scales, C, ldc, Beta, workspace, ThreadPool);
            return;
        }
        Fp8HalfWorkspace workspace;
        Fp8HalfSVGemm(M, N, K, A, lda, B, QuantType, Scales, C, ldc, Beta, workspace, ThreadPool);
        return;
    }

    if (UsesDecodedGemm(QuantType)) {
        DecodedGemmWorkspace workspace;
        DecodedSVGemm(M, N, K, A, lda, B, QuantType, Scales, C, ldc, Beta, workspace, ThreadPool);
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

    if (UsesDecodedGemm(QuantType) || UsesNativeInt8Gemm(QuantType) ||
        UsesNativeInt4Gemm(QuantType) || UsesFp8HalfGemm(QuantType)) {
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
