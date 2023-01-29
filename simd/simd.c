#include "dpengine/pixels.h"

#include <inttypes.h>
#include <stddef.h>
#include <stdio.h>

#if defined(__clang__) && defined(_MSC_VER)
// If clang is used on windows, it'll skip instrinsic headers if the corresponding macros are not defined
#    ifndef __SSE3__
#        define __SSE3__
#        define UNDEF__SSE3__
#    endif
#    ifndef __SSSE3__
#        define __SSSE3__
#        define UNDEF__SSSE3__
#    endif
#    ifndef __SSE4_1__
#        define __SSE4_1__
#        define UNDEF__SSE4_1__
#    endif
#    ifndef __SSE4_2__
#        define __SSE4_2__
#        define UNDEF__SSE4_2__
#    endif
#    ifndef __AVX__
#        define __AVX__
#        define UNDEF__AVX__
#    endif
#    ifndef __AVX2__
#        define __AVX2__
#        define UNDEF__AVX2__
#    endif

#    include <x86intrin.h>

#    ifdef UNDEF__SSE3__
#        undef __SSE3__
#        undef UNDEF__SSE3__
#    endif
#    ifdef UNDEF__SSSE3__
#        undef __SSSE3__
#        undef UNDEF__SSSE3__
#    endif
#    ifdef UNDEF__SSE4_1__
#        undef __SSE4_1__
#        undef UNDEF__SSE4_1__
#    endif
#    ifdef UNDEF__SSE4_2__
#        undef __SSE4_2__
#        undef UNDEF__SSE4_2__
#    endif
#    ifdef UNDEF__AVX__
#        undef __AVX__
#        undef UNDEF__AVX__
#    endif
#    ifdef UNDEF__AVX2__
#        undef __AVX2__
#        undef UNDEF__AVX2__
#    endif
#elif defined(__clang__) || defined(__GNUC__)
#    include <x86intrin.h>
#else
#    include <intrin.h>
#endif


#if defined(__clang__) || defined(__GNUC__)
#    define ALWAYS_INLINE static inline __attribute__((__always_inline__))
#elif defined(_MSC_VER)
#    define ALWAYS_INLINE static inline __forceinline
#elif
#    define ALWAYS_INLINE
#endif

#define DO_PRAGMA_(x) _Pragma(#x)
#define DO_PRAGMA(x)  DO_PRAGMA_(x)

#if defined(__clang__)
#    define ENABLE_TARGET_BEGIN(TARGET) DO_PRAGMA(clang attribute push(__attribute__((target(TARGET))), apply_to = function))
#    define ENABLE_TARGET_END           _Pragma("clang attribute pop")
#elif defined(__GNUC__)
#    define ENABLE_TARGET_BEGIN(TARGET) _Pragma("GCC push_options") DO_PRAGMA(GCC target(TARGET))
#    define ENABLE_TARGET_END           _Pragma("GCC pop_options");
#else
#    define ENABLE_TARGET_BEGIN(TARGET)
#    define ENABLE_TARGET_END
#endif

// SSE
ENABLE_TARGET_BEGIN("sse4.2")
ALWAYS_INLINE void load_shuffle_unpack_and(DP_Pixel15 src[4], __m128i *out_blue, __m128i *out_green, __m128i *out_red, __m128i *out_alpha)
{
    __m128i source1 = _mm_load_si128((__m128i *)src);
    __m128i source2 = _mm_load_si128((__m128i *)src + 1);

    __m128i shuffled1 = _mm_shuffle_epi32(source1, _MM_SHUFFLE(3, 1, 2, 0));
    __m128i shuffled2 = _mm_shuffle_epi32(source2, _MM_SHUFFLE(3, 1, 2, 0));

    __m128i blue_green = _mm_unpacklo_epi64(shuffled1, shuffled2);
    __m128i red_alpha = _mm_unpackhi_epi64(shuffled1, shuffled2);

    *out_blue = _mm_blend_epi16(blue_green, _mm_set1_epi32(0), 170);
    *out_green = _mm_srli_epi32(blue_green, 16);
    *out_red = _mm_blend_epi16(blue_green, _mm_set1_epi32(0), 170);
    *out_alpha = _mm_srli_epi32(red_alpha, 16);
}


ALWAYS_INLINE void store_shift_blend_unpack(__m128i blue, __m128i green, __m128i red, __m128i alpha, DP_Pixel15 dest[4])
{

    __m128i blue_green = _mm_blend_epi16(blue, _mm_slli_si128(green, 2), 170);
    __m128i red_alpha = _mm_blend_epi16(red, _mm_slli_si128(alpha, 2), 170);

    __m128i hi = _mm_unpackhi_epi32(blue_green, red_alpha);
    __m128i lo = _mm_unpacklo_epi32(blue_green, red_alpha);

    _mm_store_si128((__m128i *)dest, lo);
    _mm_store_si128((__m128i *)dest + 1, hi);
}

static __m128i mul(__m128i a, __m128i b)
{
    __m128i t = _mm_mullo_epi32(a, b);
    __m128i t2 = _mm_srli_epi32(t, 15);
    return t2;
}

void DP_blend_pixels_simd(DP_Pixel15 *dst, DP_Pixel15 *src, int pixel_count,
                          uint16_t opacity, int blend_mode)
{
    __m128i o = _mm_set1_epi32(opacity); // o = opactity

    // 4 pixels are loaded at a time
    for (size_t i = 0; i < pixel_count; i += 4) {
        // load
        __m128i srcB, srcG, srcR, srcA;
        load_shuffle_unpack_and(&src[i], &srcB, &srcG, &srcR, &srcA);

        __m128i dstB, dstG, dstR, dstA;
        load_shuffle_unpack_and(&dst[i], &dstB, &dstG, &dstR, &dstA);

        // Normal blend
        __m128i srcAO = mul(srcA, o);
        __m128i as1 = _mm_sub_epi32(_mm_set1_epi32(1 << 15), srcAO); // as1 = 1 - srcA * o

        dstB = _mm_add_epi32(mul(dstB, as1), mul(srcB, o)); // dstB = (dstB * as1) + (srcB * o)
        dstG = _mm_add_epi32(mul(dstG, as1), mul(srcG, o)); // dstG = (dstG * as1) + (srcG * o)
        dstR = _mm_add_epi32(mul(dstR, as1), mul(srcR, o)); // dstR = (dstR * as1) + (srcR * o)
        dstA = _mm_add_epi32(mul(dstA, as1), srcAO);        // dstA = (dstA * as1) + (srcA * o)

        // store
        store_shift_blend_unpack(dstB, dstG, dstR, dstA, &dst[i]);
    }
}

void DP_blend_pixels_simd_decompiled(
    DP_Pixel15 *dst,
    DP_Pixel15 *src,
    int pixel_count,
    uint16_t opacity,
    int blend_mode)
{
    __m128i v5;    // xmm10
    uint64_t v6;   // rcx
    __m128i si128; // xmm8
    __m128i v8;    // xmm5
    __m128i v9;    // xmm0
    __m128i v10;   // xmm3
    __m128i v11;   // xmm5
    __m128i v12;   // xmm4
    __m128i v13;   // xmm7
    __m128i v14;   // xmm1
    __m128i v15;   // xmm0
    __m128i v16;   // xmm7
    __m128i v17;   // xmm6
    __m128i v18;   // xmm2
    __m128i v19;   // xmm3
    __m128i v20;   // xmm4
    __m128i v21;   // xmm0
    __m128i v22;   // xmm3

    if (pixel_count) {
        v5 = _mm_shuffle_epi32(_mm_cvtsi32_si128(opacity), 0);
        v6 = 0LL;
        si128 = _mm_set1_epi32(1 << 15);
        do {
            v8 = _mm_shuffle_epi32(*(__m128i *)&src[v6].b, 216);
            v9 = _mm_shuffle_epi32(*(__m128i *)&src[v6 + 2].b, 216);
            v10 = _mm_unpacklo_epi64(v8, v9);
            v11 = _mm_unpackhi_epi64(v8, v9);
            v12 = _mm_srli_epi32(v10, 0x10u);
            v13 = _mm_shuffle_epi32(*(__m128i *)&dst[v6].b, 216);
            v14 = _mm_shuffle_epi32(*(__m128i *)&dst[v6 + 2].b, 216);
            v15 = _mm_unpacklo_epi64(v13, v14);
            v16 = _mm_unpackhi_epi64(v13, v14);
            v17 = _mm_srli_epi32(_mm_mullo_epi32(_mm_srli_epi32(v11, 0x10u), v5), 0xFu);
            v18 = _mm_sub_epi32(si128, v17);
            v19 = _mm_packus_epi32(
                _mm_add_epi32(
                    _mm_srli_epi32(_mm_mullo_epi32(_mm_blend_epi16(v10, _mm_set1_epi32(0), 170), v5), 0xFu),
                    _mm_srli_epi32(_mm_mullo_epi32(_mm_blend_epi16(v15, _mm_set1_epi32(0), 170), v18), 0xFu)),
                _mm_add_epi32(
                    _mm_srli_epi32(_mm_mullo_epi32(_mm_blend_epi16(v11, _mm_set1_epi32(0), 170), v5), 0xFu),
                    _mm_srli_epi32(_mm_mullo_epi32(_mm_blend_epi16(v16, _mm_set1_epi32(0), 170), v18), 0xFu)));
            v20 = _mm_packus_epi32(
                _mm_add_epi32(
                    _mm_srli_epi32(_mm_mullo_epi32(v12, v5), 0xFu),
                    _mm_srli_epi32(_mm_mullo_epi32(_mm_srli_epi32(v15, 0x10u), v18), 0xFu)),
                _mm_add_epi32(_mm_srli_epi32(_mm_mullo_epi32(v18, _mm_srli_epi32(v16, 0x10u)), 0xFu), v17));
            v21 = _mm_unpackhi_epi16(v19, v20);
            v22 = _mm_unpacklo_epi16(v19, v20);
            *(__m128i *)&dst[v6].b = _mm_unpacklo_epi32(v22, v21);
            *(__m128i *)&dst[v6 + 2].b = _mm_unpackhi_epi32(v22, v21);
            v6 += 4LL;
        } while (v6 < pixel_count);
    }
}
ENABLE_TARGET_END

// AVX
ENABLE_TARGET_BEGIN("avx2")
ALWAYS_INLINE void load256_permute(DP_Pixel15 src[8], __m256i *out_blue, __m256i *out_green, __m256i *out_red, __m256i *out_alpha)
{
    __m256i source1 = _mm256_loadu_si256((__m256i *)src);
    __m256i source2 = _mm256_loadu_si256((__m256i *)src + 1);

    __m256i shuffled1 = _mm256_permutevar8x32_epi32(source1, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));
    __m256i shuffled2 = _mm256_permutevar8x32_epi32(source2, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7));

    __m256i blue_green = _mm256_permute2x128_si256(shuffled1, shuffled2, 32);
    __m256i red_alpha = _mm256_permute2x128_si256(shuffled1, shuffled2, 49);

    *out_blue = _mm256_blend_epi16(blue_green, _mm256_set1_epi32(0), 170);
    *out_green = _mm256_srli_epi32(blue_green, 16);
    *out_red = _mm256_blend_epi16(red_alpha, _mm256_set1_epi32(0), 170);
    *out_alpha = _mm256_srli_epi32(red_alpha, 16);
}

ALWAYS_INLINE void store256_permute(__m256i blue, __m256i green, __m256i red, __m256i alpha, DP_Pixel15 dest[8])
{
    __m256i blue_green = _mm256_blend_epi16(blue, _mm256_slli_si256(green, 2), 170);
    __m256i red_alpha = _mm256_blend_epi16(red, _mm256_slli_si256(alpha, 2), 170);

    __m256i hi = _mm256_permute2x128_si256(blue_green, red_alpha, 32);
    __m256i lo = _mm256_permute2x128_si256(blue_green, red_alpha, 49);

    __m256i shuffled1 = _mm256_permutevar8x32_epi32(hi, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
    __m256i shuffled2 = _mm256_permutevar8x32_epi32(lo, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));

    _mm256_storeu_si256((__m256i *)dest, shuffled1);
    _mm256_storeu_si256((__m256i *)dest + 1, shuffled2);
}

ALWAYS_INLINE void load256_unpack(DP_Pixel15 src[8], __m256i *out_blue, __m256i *out_green, __m256i *out_red, __m256i *out_alpha)
{
    __m256i source1 = _mm256_loadu_si256((__m256i *)src);
    __m256i source2 = _mm256_loadu_si256((__m256i *)src + 1);

    __m256i hi = _mm256_unpackhi_epi32(source1, source2);
    __m256i lo = _mm256_unpacklo_epi32(source1, source2);

    __m256i blue_green = _mm256_unpacklo_epi64(lo, hi);
    __m256i red_alpha = _mm256_unpackhi_epi64(lo, hi);

    *out_blue = _mm256_blend_epi16(blue_green, _mm256_set1_epi32(0), 170);
    *out_green = _mm256_srli_epi32(blue_green, 16);
    *out_red = _mm256_blend_epi16(red_alpha, _mm256_set1_epi32(0), 170);
    *out_alpha = _mm256_srli_epi32(red_alpha, 16);
}

ALWAYS_INLINE void store256_unpack(__m256i blue, __m256i green, __m256i red, __m256i alpha, DP_Pixel15 dest[8])
{
    __m256i blue_green = _mm256_blend_epi16(blue, _mm256_slli_si256(green, 2), 170);
    __m256i red_alpha = _mm256_blend_epi16(red, _mm256_slli_si256(alpha, 2), 170);

    __m256i hi = _mm256_unpackhi_epi32(blue_green, red_alpha);
    __m256i lo = _mm256_unpacklo_epi32(blue_green, red_alpha);

    __m256i source1 = _mm256_unpacklo_epi64(lo, hi);
    __m256i source2 = _mm256_unpackhi_epi64(lo, hi);

    _mm256_storeu_si256((__m256i *)dest, source1);
    _mm256_storeu_si256((__m256i *)dest + 1, source2);
}

static __m256i mul256(__m256i a, __m256i b)
{
    return _mm256_srli_epi32(_mm256_mullo_epi32(a, b), 15);
}

void DP_blend_pixels_simd256(DP_Pixel15 *dst, DP_Pixel15 *src, int pixel_count,
                             uint16_t opacity, int blend_mode)
{
    __m256i o = _mm256_set1_epi32(opacity); // o = opactity

    // 4 pixels are loaded at a time
    for (size_t i = 0; i < pixel_count; i += 8) {
        // load
        __m256i srcB, srcG, srcR, srcA;
        load256_permute(&src[i], &srcB, &srcG, &srcR, &srcA);

        __m256i dstB, dstG, dstR, dstA;
        load256_permute(&dst[i], &dstB, &dstG, &dstR, &dstA);

        // Normal blend
        __m256i srcAO = mul256(srcA, o);
        __m256i as1 = _mm256_sub_epi32(_mm256_set1_epi32(1 << 15), srcAO); // as1 = 1 - srcA * o

        dstB = _mm256_add_epi32(mul256(dstB, as1), mul256(srcB, o)); // dstB = (dstB * as1) + (srcB * o)
        dstG = _mm256_add_epi32(mul256(dstG, as1), mul256(srcG, o)); // dstG = (dstG * as1) + (srcG * o)
        dstR = _mm256_add_epi32(mul256(dstR, as1), mul256(srcR, o)); // dstR = (dstR * as1) + (srcR * o)
        dstA = _mm256_add_epi32(mul256(dstA, as1), srcAO);           // dstA = (dstA * as1) + (srcA * o)

        // store
        store256_permute(dstB, dstG, dstR, dstA, &dst[i]);
    }
}

void DP_blend_pixels_simd256_unpack(DP_Pixel15 *dst, DP_Pixel15 *src, int pixel_count,
                                    uint16_t opacity, int blend_mode)
{
    __m256i o = _mm256_set1_epi32(opacity); // o = opactity

    // 4 pixels are loaded at a time
    for (size_t i = 0; i < pixel_count; i += 8) {
        // load
        __m256i srcB, srcG, srcR, srcA;
        load256_unpack(&src[i], &srcB, &srcG, &srcR, &srcA);

        __m256i dstB, dstG, dstR, dstA;
        load256_unpack(&dst[i], &dstB, &dstG, &dstR, &dstA);

        // Normal blend
        __m256i srcAO = mul256(srcA, o);
        __m256i as1 = _mm256_sub_epi32(_mm256_set1_epi32(1 << 15), srcAO); // as1 = 1 - srcA * o

        dstB = _mm256_add_epi32(mul256(dstB, as1), mul256(srcB, o)); // dstB = (dstB * as1) + (srcB * o)
        dstG = _mm256_add_epi32(mul256(dstG, as1), mul256(srcG, o)); // dstG = (dstG * as1) + (srcG * o)
        dstR = _mm256_add_epi32(mul256(dstR, as1), mul256(srcR, o)); // dstR = (dstR * as1) + (srcR * o)
        dstA = _mm256_add_epi32(mul256(dstA, as1), srcAO);           // dstA = (dstA * as1) + (srcA * o)

        // store
        store256_unpack(dstB, dstG, dstR, dstA, &dst[i]);
    }
}


typedef struct PixelFloatSoA {
    float *b;
    float *g;
    float *r;
    float *a;
} PixelFloatSoA;

void DP_blend_pixels_simd_float_soa(PixelFloatSoA *dst, PixelFloatSoA *src, int pixel_count,
                                    uint16_t opacity, int blend_mode)
{
    __m128 o = _mm_set1_ps((float)opacity / (float)(1 << 15)); // o = opactity / (1 << 15)

    // 4 pixels are loaded at a time
    for (size_t i = 0; i < pixel_count; i += 4) {

        // load
        __m128 srcB = _mm_load_ps((&src->b[i]));
        __m128 srcG = _mm_load_ps((&src->g[i]));
        __m128 srcR = _mm_load_ps((&src->r[i]));
        __m128 srcA = _mm_load_ps((&src->a[i]));

        __m128 dstB = _mm_load_ps((&dst->b[i]));
        __m128 dstG = _mm_load_ps((&dst->g[i]));
        __m128 dstR = _mm_load_ps((&dst->r[i]));
        __m128 dstA = _mm_load_ps((&dst->a[i]));

        // Normal blend
        __m128 as1 = _mm_sub_ps(_mm_set1_ps(1), _mm_mul_ps(srcA, o)); // as1 = 1 - dstA * o

        dstB = _mm_add_ps(_mm_mul_ps(dstB, as1), _mm_mul_ps(srcB, o)); // dstB = (dstB * as1) + (srcB * o)
        dstG = _mm_add_ps(_mm_mul_ps(dstG, as1), _mm_mul_ps(srcG, o)); // dstG = (dstG * as1) + (srcG * o)
        dstR = _mm_add_ps(_mm_mul_ps(dstR, as1), _mm_mul_ps(srcR, o)); // dstR = (dstR * as1) + (srcR * o)
        dstA = _mm_add_ps(_mm_mul_ps(dstA, as1), _mm_mul_ps(srcA, o)); // dstA = (dstA * as1) + (srcA * o)

        // store
        _mm_store_ps(&dst->b[i], dstB);
        _mm_store_ps(&dst->g[i], dstG);
        _mm_store_ps(&dst->r[i], dstR);
        _mm_store_ps(&dst->a[i], dstA);
    }
}

typedef struct Pixel15SoA {
    uint32_t *b;
    uint32_t *g;
    uint32_t *r;
    uint32_t *a;
} Pixel15SoA;

void DP_blend_pixels_simd_15_soa(Pixel15SoA *dst, Pixel15SoA *src, int pixel_count,
                                 uint16_t opacity, int blend_mode)
{
    __m128i o = _mm_set1_epi32(opacity); // o = opactity

    // 4 pixels are loaded at a time
    for (size_t i = 0; i < pixel_count; i += 4) {

        // load
        __m128i srcB = _mm_load_si128((__m128i *)(&src->b[i]));
        __m128i srcG = _mm_load_si128((__m128i *)(&src->g[i]));
        __m128i srcR = _mm_load_si128((__m128i *)(&src->r[i]));
        __m128i srcA = _mm_load_si128((__m128i *)(&src->a[i]));

        __m128i dstB = _mm_load_si128((__m128i *)(&dst->b[i]));
        __m128i dstG = _mm_load_si128((__m128i *)(&dst->g[i]));
        __m128i dstR = _mm_load_si128((__m128i *)(&dst->r[i]));
        __m128i dstA = _mm_load_si128((__m128i *)(&dst->a[i]));

        // Normal blend
        __m128i as1 = _mm_sub_epi32(_mm_set1_epi32(1 << 15), mul(srcA, o)); // as1 = 1 - srcA * o

        dstB = _mm_add_epi32(mul(dstB, as1), mul(srcB, o)); // dstB = (dstB * as1) + (srcB * o)
        dstG = _mm_add_epi32(mul(dstG, as1), mul(srcG, o)); // dstG = (dstG * as1) + (srcG * o)
        dstR = _mm_add_epi32(mul(dstR, as1), mul(srcR, o)); // dstR = (dstR * as1) + (srcR * o)
        dstA = _mm_add_epi32(mul(dstA, as1), mul(srcA, o)); // dstA = (dstA * as1) + (srcA * o)

        // store
        _mm_store_si128((__m128i *)&dst->b[i], dstB);
        _mm_store_si128((__m128i *)&dst->g[i], dstG);
        _mm_store_si128((__m128i *)&dst->r[i], dstR);
        _mm_store_si128((__m128i *)&dst->a[i], dstA);
    }
}
ENABLE_TARGET_END
