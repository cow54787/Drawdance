#include <emmintrin.h>
#include <immintrin.h>
#include <inttypes.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <tmmintrin.h>
#include <xmmintrin.h>

typedef struct DP_Pixel15 {
    uint16_t b, g, r, a;
} DP_Pixel15;

#define DP_BLEND_MODE_MAX 255

typedef enum DP_BlendMode {
    DP_BLEND_MODE_ERASE = 0,
    DP_BLEND_MODE_NORMAL,
    DP_BLEND_MODE_MULTIPLY,
    DP_BLEND_MODE_DIVIDE,
    DP_BLEND_MODE_BURN,
    DP_BLEND_MODE_DODGE,
    DP_BLEND_MODE_DARKEN,
    DP_BLEND_MODE_LIGHTEN,
    DP_BLEND_MODE_SUBTRACT,
    DP_BLEND_MODE_ADD,
    DP_BLEND_MODE_RECOLOR,
    DP_BLEND_MODE_BEHIND,
    DP_BLEND_MODE_COLOR_ERASE,
    DP_BLEND_MODE_SCREEN,
    DP_BLEND_MODE_NORMAL_AND_ERASER,
    DP_BLEND_MODE_LUMINOSITY_SHINE_SAI,
    DP_BLEND_MODE_OVERLAY,
    DP_BLEND_MODE_HARD_LIGHT,
    DP_BLEND_MODE_SOFT_LIGHT,
    DP_BLEND_MODE_LINEAR_BURN,
    DP_BLEND_MODE_LINEAR_LIGHT,
    DP_BLEND_MODE_HUE,
    DP_BLEND_MODE_SATURATION,
    DP_BLEND_MODE_LUMINOSITY,
    DP_BLEND_MODE_COLOR,
    DP_BLEND_MODE_LAST_EXCEPT_REPLACE, // Put new blend modes before this value.
    DP_BLEND_MODE_REPLACE = DP_BLEND_MODE_MAX,
    DP_BLEND_MODE_COUNT,
} DP_BlendMode;


typedef uint_fast32_t Fix15;
typedef int_fast32_t IFix15;

#define DP_ASSERT

#define DP_BIT15 (1 << 15)

#define BIT15_U16    ((uint16_t)DP_BIT15)
#define BIT15_FLOAT  ((float)DP_BIT15)
#define BIT15_DOUBLE ((double)DP_BIT15)
#define BIT15_FIX    ((Fix15)DP_BIT15)
#define BIT15_IFIX   ((IFix15)DP_BIT15)

#define FIX_1         ((Fix15)1)
#define FIX_2         ((Fix15)2)
#define FIX_4         ((Fix15)4)
#define FIX_12        ((Fix15)12)
#define FIX_15        ((Fix15)15)
#define FIX_16        ((Fix15)16)
#define BIT15_INC_FIX ((Fix15)(DP_BIT15 + 1))

typedef struct BGR15 {
    Fix15 b, g, r;
} BGR15;

typedef struct IBGR15 {
    IFix15 b, g, r;
} IBGR15;

typedef struct BGRA15 {
    union {
        BGR15 bgr;
        struct {
            Fix15 b, g, r;
        };
    };
    Fix15 a;
} BGRA15;


static Fix15 to_fix(uint16_t x)
{
    return (Fix15)x;
}

static uint16_t from_fix(Fix15 x)
{
    DP_ASSERT(x <= BIT15_FIX);
    return (uint16_t)x;
}

// Adapted from MyPaint, see license above.
static Fix15 fix15_mul(Fix15 a, Fix15 b)
{
    return (a * b) >> FIX_15;
}


static BGR15 to_bgr(DP_Pixel15 pixel)
{
    return (BGR15){
        .b = to_fix(pixel.b),
        .g = to_fix(pixel.g),
        .r = to_fix(pixel.r),
    };
}

static BGRA15 to_bgra(DP_Pixel15 pixel)
{
    return (BGRA15){
        .bgr = to_bgr(pixel),
        .a = to_fix(pixel.a),
    };
}

static DP_Pixel15 from_bgra(BGRA15 bgra)
{
    return (DP_Pixel15){
        .b = from_fix(bgra.b),
        .g = from_fix(bgra.g),
        .r = from_fix(bgra.r),
        .a = from_fix(bgra.a),
    };
}


uint16_t DP_fix15_mul(uint16_t a, uint16_t b)
{
    return from_fix(fix15_mul(to_fix(a), to_fix(b)));
}


static BGRA15 blend_normal(BGR15 cb, BGR15 cs, Fix15 ab, Fix15 as, Fix15 o)
{
    Fix15 as1 = BIT15_FIX - fix15_mul(as, o);
    return (BGRA15){
        .b = fix15_mul(cb.b, as1) + fix15_mul(cs.b, o),
        .g = fix15_mul(cb.g, as1) + fix15_mul(cs.g, o),
        .r = fix15_mul(cb.r, as1) + fix15_mul(cs.r, o),
        .a = fix15_mul(ab, as1) + fix15_mul(as, o),
    };
}


#define FOR_PIXEL(DST, SRC, PIXEL_COUNT, I, BLOCK)            \
    do {                                                      \
        for (int I = 0; I < PIXEL_COUNT; ++I, ++DST, ++SRC) { \
            BLOCK                                             \
        }                                                     \
    } while (0)

static void blend_pixels_alpha_op(DP_Pixel15 *dst, DP_Pixel15 *src,
                                  int pixel_count, Fix15 opacity,
                                  BGRA15 (*op)(BGR15, BGR15, Fix15, Fix15,
                                               Fix15))
{
    FOR_PIXEL(dst, src, pixel_count, i, {
        BGRA15 b = to_bgra(*dst);
        BGRA15 s = to_bgra(*src);
        *dst = from_bgra(op(b.bgr, s.bgr, b.a, s.a, opacity));
    });
}

void DP_blend_pixels(DP_Pixel15 *dst, DP_Pixel15 *src, int pixel_count,
                     uint16_t opacity, int blend_mode)
{
    switch (blend_mode) {
    // Alpha-affecting blend modes.
    case DP_BLEND_MODE_NORMAL:
        blend_pixels_alpha_op(dst, src, pixel_count, to_fix(opacity),
                              blend_normal);
        break;
    }
}


// SSE
static inline __attribute__((always_inline)) void load_shuffle_unpack_and(DP_Pixel15 src[4], __m128i *out_blue, __m128i *out_green, __m128i *out_red, __m128i *out_alpha)
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


static inline __attribute__((always_inline)) void store_shift_blend_unpack(__m128i blue, __m128i green, __m128i red, __m128i alpha, DP_Pixel15 dest[4])
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

// AVX
static inline __attribute__((always_inline)) void load256_permute(DP_Pixel15 src[8], __m256i *out_blue, __m256i *out_green, __m256i *out_red, __m256i *out_alpha)
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

static inline __attribute__((always_inline)) void store256_permute(__m256i blue, __m256i green, __m256i red, __m256i alpha, DP_Pixel15 dest[8])
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

static inline __attribute__((always_inline)) void load256_unpack(DP_Pixel15 src[8], __m256i *out_blue, __m256i *out_green, __m256i *out_red, __m256i *out_alpha)
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

static inline __attribute__((always_inline)) void store256_unpack(__m256i blue, __m256i green, __m256i red, __m256i alpha, DP_Pixel15 dest[8])
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
