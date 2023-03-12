#include "bench_common.hpp"

extern "C" {
#include "dpcommon/cpu.h"
#include "dpcommon/input.h"
#include "dpcommon/output.h"
#include "dpcommon/perf.h"
#include "dpengine/image.h"
#include "dpengine/pixels.h"
#include "dpengine/tile.h"
#include "dpmsg/blend_mode.h"
}

#include <benchmark/benchmark.h>

#pragma region seq
static void calculate_rr_mask_row_seq(float *rr_mask_row, int start_x, int yp,
                                      int count, float radius,
                                      float aspect_ratio, float sn, float cs,
                                      float one_over_radius2)
{
    for (int xp = start_x; xp < start_x + count; ++xp) {
        float yy = (float)yp + 0.5f - radius;
        float xx = (float)xp + 0.5f - radius;
        float yyr = (yy * cs - xx * sn) * aspect_ratio;
        float xxr = yy * sn + xx * cs;
        float rr = (yyr * yyr + xxr * xxr) * one_over_radius2;

        rr_mask_row[xp] = rr;
    }
}

static void calculate_rr_mask_seq(float *rr_mask, int idia, float radius,
                                  float aspect_ratio, float sn, float cs,
                                  float one_over_radius2)
{
    for (int yp = 0; yp < idia; ++yp) {
        calculate_rr_mask_row_seq(&rr_mask[yp * idia], 0, yp, idia, radius,
                                  aspect_ratio, sn, cs, one_over_radius2);
    }
}
#pragma endregion


#pragma region avx
DP_TARGET_BEGIN("avx")
static void calculate_rr_mask_row_avx(float *rr_mask_row, int start_x,
                                      int yp_int, int count, float radius,
                                      float aspect_ratio_float, float sn_float,
                                      float cs_float,
                                      float one_over_radius2_float)
{
    DP_ASSERT(count % 8 == 0);

    // Refer to calculate_rr_mask_row for the formulas

    __m256 half_minus_radius =
        _mm256_sub_ps(_mm256_set1_ps(0.5f), _mm256_set1_ps((float)radius));

    __m256 aspect_ratio = _mm256_set1_ps(aspect_ratio_float);
    __m256 sn = _mm256_set1_ps(sn_float);
    __m256 cs = _mm256_set1_ps(cs_float);
    __m256 one_over_radius2 = _mm256_set1_ps(one_over_radius2_float);

    __m256 yp = _mm256_set1_ps((float)yp_int);
    __m256 yy = _mm256_add_ps(yp, half_minus_radius);

    __m256 xp = _mm256_add_ps(
        _mm256_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f),
        _mm256_set1_ps((float)start_x));

    for (int i = start_x; i < start_x + count; i += 8) {
        __m256 xx = _mm256_add_ps(xp, half_minus_radius);
        __m256 yyr = _mm256_mul_ps(
            _mm256_sub_ps(_mm256_mul_ps(yy, cs), _mm256_mul_ps(xx, sn)),
            aspect_ratio);

        __m256 xxr =
            _mm256_add_ps(_mm256_mul_ps(yy, sn), _mm256_mul_ps(xx, cs));

        __m256 rr = _mm256_mul_ps(
            _mm256_add_ps(_mm256_mul_ps(yyr, yyr), _mm256_mul_ps(xxr, xxr)),
            one_over_radius2);

        _mm256_storeu_ps(&rr_mask_row[i], rr);

        xp = _mm256_add_ps(xp, _mm256_set1_ps(8.0f));
    }
    _mm256_zeroupper();
}

static void calculate_rr_mask_avx_rowbyrow(float *rr_mask, int idia,
                                           float radius, float aspect_ratio,
                                           float sn, float cs,
                                           float one_over_radius2)
{
    for (int yp = 0; yp < idia; ++yp) {
        int xp = 0;
        int remaining = idia;

        int remaining_after_avx_width = remaining % 8;
        int avx_width = remaining - remaining_after_avx_width;

        calculate_rr_mask_row_avx(&rr_mask[yp * idia], xp, yp, avx_width,
                                  radius, aspect_ratio, sn, cs,
                                  one_over_radius2);

        remaining -= avx_width;
        xp += avx_width;


        calculate_rr_mask_row_seq(&rr_mask[yp * idia], xp, yp, remaining,
                                  radius, aspect_ratio, sn, cs,
                                  one_over_radius2);
    }
}

static void calculate_rr_mask_row_avx_2(float *rr_mask_row, int start_x,
                                        int yp_int, int count, float radius,
                                        float aspect_ratio_float,
                                        float sn_float, float cs_float,
                                        float one_over_radius2_float)
{
    DP_ASSERT(count % 8 == 0);

    // Refer to calculate_rr_mask_row for the formulas

    __m256 half_minus_radius =
        _mm256_sub_ps(_mm256_set1_ps(0.5f), _mm256_set1_ps((float)radius));

    __m256 aspect_ratio = _mm256_set1_ps(aspect_ratio_float);
    __m256 sn = _mm256_set1_ps(sn_float);
    __m256 cs = _mm256_set1_ps(cs_float);
    __m256 one_over_radius2 = _mm256_set1_ps(one_over_radius2_float);

    __m256 yp = _mm256_set1_ps((float)yp_int);
    __m256 yy = _mm256_add_ps(yp, half_minus_radius);
    __m256 yy_times_cs = _mm256_mul_ps(yy, cs);
    __m256 yy_times_ss = _mm256_mul_ps(yy, sn);

    __m256 xp = _mm256_add_ps(
        _mm256_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f),
        _mm256_set1_ps((float)start_x));

    for (int i = start_x; i < start_x + count; i += 8) {
        __m256 xx = _mm256_add_ps(xp, half_minus_radius);
        __m256 yyr = _mm256_mul_ps(
            _mm256_sub_ps(yy_times_cs, _mm256_mul_ps(xx, sn)), aspect_ratio);

        __m256 xxr = _mm256_add_ps(yy_times_ss, _mm256_mul_ps(xx, cs));

        __m256 rr = _mm256_mul_ps(
            _mm256_add_ps(_mm256_mul_ps(yyr, yyr), _mm256_mul_ps(xxr, xxr)),
            one_over_radius2);

        _mm256_storeu_ps(&rr_mask_row[i], rr);

        xp = _mm256_add_ps(xp, _mm256_set1_ps(8.0f));
    }
    _mm256_zeroupper();
}

static void calculate_rr_mask_avx_rowbyrow2(float *rr_mask, int idia,
                                            float radius, float aspect_ratio,
                                            float sn, float cs,
                                            float one_over_radius2)
{
    for (int yp = 0; yp < idia; ++yp) {
        int xp = 0;
        int remaining = idia;

        int remaining_after_avx_width = remaining % 8;
        int avx_width = remaining - remaining_after_avx_width;

        calculate_rr_mask_row_avx_2(&rr_mask[yp * idia], xp, yp, avx_width,
                                    radius, aspect_ratio, sn, cs,
                                    one_over_radius2);

        remaining -= avx_width;
        xp += avx_width;


        calculate_rr_mask_row_seq(&rr_mask[yp * idia], xp, yp, remaining,
                                  radius, aspect_ratio, sn, cs,
                                  one_over_radius2);
    }
}

static void calculate_rr_mask_avx_single_loop(float *rr_mask, int idia,
                                              float radius_float,
                                              float aspect_ratio_float,
                                              float sn_float, float cs_float,
                                              float one_over_radius2_float)
{
    int count = idia * idia;
    int remaining_after_avx = count % 8;
    int avx_count = count - remaining_after_avx;

    __m256 half_minus_radius = _mm256_sub_ps(
        _mm256_set1_ps(0.5f), _mm256_set1_ps((float)radius_float));

    __m256 aspect_ratio = _mm256_set1_ps(aspect_ratio_float);
    __m256 sn = _mm256_set1_ps(sn_float);
    __m256 cs = _mm256_set1_ps(cs_float);
    __m256 one_over_radius2 = _mm256_set1_ps(one_over_radius2_float);

    __m256 xp = _mm256_setr_ps(0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f);
    __m256 yp = _mm256_set1_ps(0.0f);

    for (int i = 0; i < avx_count; i += 8) {

        // Check if Y needs to be increased and X wrap around
        __m256 mask =
            _mm256_cmp_ps(xp, _mm256_set1_ps((float)idia), _CMP_GE_OS);
        if (_mm256_movemask_ps(mask)) {
            yp = _mm256_add_ps(yp, _mm256_and_ps(mask, _mm256_set1_ps(1.0f)));
            xp = _mm256_sub_ps(
                xp, _mm256_and_ps(mask, _mm256_set1_ps((float)idia)));
        }

        __m256 xx = _mm256_add_ps(xp, half_minus_radius);
        __m256 yy = _mm256_add_ps(yp, half_minus_radius);
        __m256 yyr = _mm256_mul_ps(
            _mm256_sub_ps(_mm256_mul_ps(yy, cs), _mm256_mul_ps(xx, sn)),
            aspect_ratio);

        __m256 xxr =
            _mm256_add_ps(_mm256_mul_ps(yy, sn), _mm256_mul_ps(xx, cs));

        __m256 rr = _mm256_mul_ps(
            _mm256_add_ps(_mm256_mul_ps(yyr, yyr), _mm256_mul_ps(xxr, xxr)),
            one_over_radius2);

        _mm256_store_ps(&rr_mask[i], rr);

        xp = _mm256_add_ps(xp, _mm256_set1_ps(8.0f));
    }

    // Seq cleanup
    int seq_start_y = avx_count / idia;
    int seq_start_x = avx_count % idia;
    for (int yp = seq_start_y; yp < idia; yp++) {
        for (int xp = seq_start_x; xp < idia; xp++) {
            float yy = (float)yp + 0.5f - radius_float;
            float xx = (float)xp + 0.5f - radius_float;
            float yyr = (yy * cs_float - xx * sn_float) * aspect_ratio_float;
            float xxr = yy * sn_float + xx * cs_float;
            float rr = (yyr * yyr + xxr * xxr) * one_over_radius2_float;

            rr_mask[yp * idia + xp] = rr;
        }
    }
}


DP_TARGET_END
#pragma endregion

DISABLE_OPT_BEGIN
#define BENCH_RR(name, func)                                             \
    static void name(benchmark::State &state)                            \
    {                                                                    \
        int idia = 260;                                                  \
                                                                         \
        float *rr_mask =                                                 \
            (float *)DP_malloc_simd_zeroed(idia * idia * sizeof(float)); \
                                                                         \
        float radius = (float)rand();                                    \
        float aspect_ratio = (float)rand();                              \
        float sn = (float)rand();                                        \
        float cs = (float)rand();                                        \
        float one_over_radius2 = (float)rand();                          \
                                                                         \
        for (auto _ : state) {                                           \
            benchmark::DoNotOptimize(rr_mask);                           \
            benchmark::DoNotOptimize(radius);                            \
            benchmark::DoNotOptimize(aspect_ratio);                      \
            benchmark::DoNotOptimize(sn);                                \
            benchmark::DoNotOptimize(cs);                                \
            benchmark::DoNotOptimize(one_over_radius2);                  \
                                                                         \
            func(rr_mask, idia, radius, aspect_ratio, sn, cs,            \
                 one_over_radius2);                                      \
                                                                         \
            benchmark::DoNotOptimize(rr_mask);                           \
        }                                                                \
        DP_free_simd(rr_mask);                                           \
    }                                                                    \
    BENCHMARK(name)

BENCH_RR(seq, calculate_rr_mask_seq);
BENCH_RR(avx_rowbyrow, calculate_rr_mask_avx_rowbyrow);
BENCH_RR(avx_rowbyrow2, calculate_rr_mask_avx_rowbyrow2);
BENCH_RR(avx_single_loop, calculate_rr_mask_avx_single_loop);

DISABLE_OPT_END

BENCHMARK_MAIN();

void validate()
{
    int idia = 215;

    float *rr_mask_seq =
        (float *)DP_malloc_simd_zeroed(idia * idia * sizeof(float));
    float *rr_mask_avx =
        (float *)DP_malloc_simd_zeroed(idia * idia * sizeof(float));

    float radius = (float)rand();
    float aspect_ratio = (float)rand();
    float sn = (float)rand();
    float cs = (float)rand();
    float one_over_radius2 = (float)rand();

    calculate_rr_mask_seq(rr_mask_seq, idia, radius, aspect_ratio, sn, cs,
                          one_over_radius2);
    calculate_rr_mask_avx_rowbyrow2(rr_mask_avx, idia, radius, aspect_ratio, sn,
                                    cs, one_over_radius2);

    for (int i = 0; i < idia * idia; i++) {
        if (rr_mask_seq[i] != rr_mask_avx[i]) {
            DP_panic("%f != %f (index : %d)", rr_mask_seq[i], rr_mask_avx[i],
                     i);
        }
    }

    DP_free_simd(rr_mask_seq);
    DP_free_simd(rr_mask_avx);
}
