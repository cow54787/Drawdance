
extern "C" {
#include "dpcommon/input.h"
#include "dpcommon/output.h"
#include "dpcommon/perf.h"
#include "dpengine/image.h"
#include "dpengine/pixels.h"
#include "dpengine/tile.h"
#include "dpmsg/blend_mode.h"
#include "simd.c"
}

#include <dpcommon/common.h>
#include <benchmark/benchmark.h>


static void BM_seq(benchmark::State &state)
{
    DP_Pixel15 *dst = (DP_Pixel15 *)DP_malloc(DP_TILE_LENGTH * sizeof(DP_Pixel15));
    DP_Pixel15 *src = (DP_Pixel15 *)DP_malloc(DP_TILE_LENGTH * sizeof(DP_Pixel15));

    for (auto _ : state) {
        benchmark::DoNotOptimize(src);
        benchmark::DoNotOptimize(dst);
        DP_blend_pixels(dst, src, DP_TILE_LENGTH, DP_BIT15, DP_BLEND_MODE_NORMAL);
        benchmark::DoNotOptimize(dst);
    }

    DP_free(dst);
    DP_free(src);
}
BENCHMARK(BM_seq);

static void BM_simd_shuffle(benchmark::State &state)
{
    DP_Pixel15 *dst = (DP_Pixel15 *)DP_malloc(DP_TILE_LENGTH * sizeof(DP_Pixel15));
    DP_Pixel15 *src = (DP_Pixel15 *)DP_malloc(DP_TILE_LENGTH * sizeof(DP_Pixel15));

    for (auto _ : state) {
        benchmark::DoNotOptimize(src);
        benchmark::DoNotOptimize(dst);
        DP_blend_pixels_simd(dst, src, DP_TILE_LENGTH, DP_BIT15, DP_BLEND_MODE_NORMAL);
        benchmark::DoNotOptimize(dst);
    }

    DP_free(dst);
    DP_free(src);
}
BENCHMARK(BM_simd_shuffle);

static void BM_simd_shuffle256(benchmark::State &state)
{
    DP_Pixel15 *dst = (DP_Pixel15 *)DP_malloc(DP_TILE_LENGTH * sizeof(DP_Pixel15));
    DP_Pixel15 *src = (DP_Pixel15 *)DP_malloc(DP_TILE_LENGTH * sizeof(DP_Pixel15));

    for (auto _ : state) {
        benchmark::DoNotOptimize(src);
        benchmark::DoNotOptimize(dst);
        DP_blend_pixels_simd256(dst, src, DP_TILE_LENGTH, DP_BIT15, DP_BLEND_MODE_NORMAL);
        benchmark::DoNotOptimize(dst);
    }

    DP_free(dst);
    DP_free(src);
}
BENCHMARK(BM_simd_shuffle256);

static void BM_simd_shuffle256_unpack(benchmark::State &state)
{
    DP_Pixel15 *dst = (DP_Pixel15 *)DP_malloc(DP_TILE_LENGTH * sizeof(DP_Pixel15));
    DP_Pixel15 *src = (DP_Pixel15 *)DP_malloc(DP_TILE_LENGTH * sizeof(DP_Pixel15));

    for (auto _ : state) {
        benchmark::DoNotOptimize(src);
        benchmark::DoNotOptimize(dst);
        DP_blend_pixels_simd256_unpack(dst, src, DP_TILE_LENGTH, DP_BIT15, DP_BLEND_MODE_NORMAL);
        benchmark::DoNotOptimize(dst);
    }

    DP_free(dst);
    DP_free(src);
}
BENCHMARK(BM_simd_shuffle256_unpack);

static void BM_simd_shuffle_decompiled(benchmark::State &state)
{
    DP_Pixel15 *dst = (DP_Pixel15 *)DP_malloc(DP_TILE_LENGTH * sizeof(DP_Pixel15));
    DP_Pixel15 *src = (DP_Pixel15 *)DP_malloc(DP_TILE_LENGTH * sizeof(DP_Pixel15));

    for (auto _ : state) {
        benchmark::DoNotOptimize(src);
        benchmark::DoNotOptimize(dst);
        DP_blend_pixels_simd_decompiled(dst, src, DP_TILE_LENGTH, DP_BIT15, DP_BLEND_MODE_NORMAL);
        benchmark::DoNotOptimize(dst);
    }

    DP_free(dst);
    DP_free(src);
}
BENCHMARK(BM_simd_shuffle_decompiled);

static void BM_simd_float_soa(benchmark::State &state)
{
    PixelFloatSoA src = {
        .b = (float *)DP_malloc(DP_TILE_LENGTH * sizeof(float)),
        .g = (float *)DP_malloc(DP_TILE_LENGTH * sizeof(float)),
        .r = (float *)DP_malloc(DP_TILE_LENGTH * sizeof(float)),
        .a = (float *)DP_malloc(DP_TILE_LENGTH * sizeof(float)),
    };

    PixelFloatSoA dst = {
        .b = (float *)DP_malloc(DP_TILE_LENGTH * sizeof(float)),
        .g = (float *)DP_malloc(DP_TILE_LENGTH * sizeof(float)),
        .r = (float *)DP_malloc(DP_TILE_LENGTH * sizeof(float)),
        .a = (float *)DP_malloc(DP_TILE_LENGTH * sizeof(float)),
    };

    for (auto _ : state) {
        benchmark::DoNotOptimize(src);
        benchmark::DoNotOptimize(dst);
        DP_blend_pixels_simd_float_soa(&dst, &src, DP_TILE_LENGTH, DP_BIT15, DP_BLEND_MODE_NORMAL);
        benchmark::DoNotOptimize(dst);
    }
}
BENCHMARK(BM_simd_float_soa);

static void BM_simd_15_soa(benchmark::State &state)
{
    Pixel15SoA src = {
        .b = (uint32_t *)DP_malloc(DP_TILE_LENGTH * sizeof(uint32_t)),
        .g = (uint32_t *)DP_malloc(DP_TILE_LENGTH * sizeof(uint32_t)),
        .r = (uint32_t *)DP_malloc(DP_TILE_LENGTH * sizeof(uint32_t)),
        .a = (uint32_t *)DP_malloc(DP_TILE_LENGTH * sizeof(uint32_t)),
    };

    Pixel15SoA dst = {
        .b = (uint32_t *)DP_malloc(DP_TILE_LENGTH * sizeof(uint32_t)),
        .g = (uint32_t *)DP_malloc(DP_TILE_LENGTH * sizeof(uint32_t)),
        .r = (uint32_t *)DP_malloc(DP_TILE_LENGTH * sizeof(uint32_t)),
        .a = (uint32_t *)DP_malloc(DP_TILE_LENGTH * sizeof(uint32_t)),
    };

    for (auto _ : state) {
        benchmark::DoNotOptimize(src);
        benchmark::DoNotOptimize(dst);
        DP_blend_pixels_simd_15_soa(&dst, &src, DP_TILE_LENGTH, DP_BIT15, DP_BLEND_MODE_NORMAL);
        benchmark::DoNotOptimize(dst);
    }
}
BENCHMARK(BM_simd_15_soa);

BENCHMARK_MAIN();
