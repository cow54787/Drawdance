#include "simd.c"

#include "dpcommon/input.h"
#include "dpcommon/output.h"
#include "dpcommon/perf.h"
#include "dpengine/image.h"
#include "dpengine/pixels.h"
#include "dpengine/tile.h"
#include "dpmsg/blend_mode.h"
#include <dpcommon/common.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#    include <windows.h>
#else
#    include <time.h>
#endif

#if defined(_MSC_VER)
#    define DISABLE_OPT_BEGIN _Pragma("optimize(\"\", off)")
#    define DISABLE_OPT_END   _Pragma("optimize(\"\", on)")
#elif defined(__clang__)
#    define DISABLE_OPT_BEGIN _Pragma("clang optimize off")
#    define DISABLE_OPT_END   _Pragma("clang optimize on")
#elif defined(__GNUC__)
#    define DISABLE_OPT_BEGIN _Pragma("GCC push_options") _Pragma("GCC optimize(\"O0\")")
#    define DISABLE_OPT_END   _Pragma("GCC pop_options")
#endif

static inline double clock_ms()
{
#ifdef _WIN32

    LARGE_INTEGER ticks;
    if (!QueryPerformanceCounter(&ticks)) {
        return -1;
    }

    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    return ((double)ticks.QuadPart * 1000.0 / (double)freq.QuadPart);
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return ((double)t.tv_nsec / 1000000.0) + ((double)t.tv_sec * 1000.0);
#endif
}


DP_Pixel15 *load_image(char *name, int *width, int *height)
{
    DP_Input *input = DP_file_input_new_from_path(name);
    DP_Image *img =
        DP_image_new_from_file(input, DP_IMAGE_FILE_TYPE_GUESS, NULL);

    *width = DP_image_width(img);
    *height = DP_image_height(img);

    DP_Pixel15 *out = DP_malloc(*width * *height * sizeof(DP_Pixel15));
    DP_pixels8_to_15(out, DP_image_pixels(img), *width * *height);

    DP_input_free(input);
    DP_image_free(img);

    return out;
}

void save_image(char *name, DP_Pixel15 *src, int width, int height)
{
    DP_Image *img = DP_image_new(width, height);
    DP_pixels15_to_8(DP_image_pixels(img), src, width * height);

    DP_Output *output = DP_file_output_new_from_path(name);
    DP_image_write_png(img, output);
    DP_output_free(output);
    DP_image_free(img);
}

void to_p(DP_Pixel15 *src, PixelFloatSoA *dst, int count)
{
    for (int i = 0; i < count; i++) {
        dst->b[i] = (float)src[i].b / (float)DP_BIT15;
        dst->g[i] = (float)src[i].g / (float)DP_BIT15;
        dst->r[i] = (float)src[i].r / (float)DP_BIT15;
        dst->a[i] = (float)src[i].a / (float)DP_BIT15;
    }
}

void from_p(PixelFloatSoA *src, DP_Pixel15 *dst, int count)
{
    for (int i = 0; i < count; i++) {

        dst[i].b = src->b[i] * (float)DP_BIT15;
        dst[i].g = src->g[i] * (float)DP_BIT15;
        dst[i].r = src->r[i] * (float)DP_BIT15;
        dst[i].a = src->a[i] * (float)DP_BIT15;
    }
}

typedef void (*BLEND_FUNC)(DP_Pixel15 *dst, DP_Pixel15 *src, int pixel_count,
                           uint16_t opacity, int blend_mode);

DISABLE_OPT_BEGIN
void bench_load(char *name, int it, BLEND_FUNC func)
{
    // random pixels
    DP_Pixel15 src[DP_TILE_LENGTH] = {0};
    DP_Pixel15 dst[DP_TILE_LENGTH] = {0};

    double start = clock_ms();
    for (int i = 0; i < it; i++) {
        func(dst, src, DP_TILE_LENGTH, DP_BIT15, DP_BLEND_MODE_NORMAL);
    }
    double end = clock_ms();

    printf("it: %d %s: %fms\n", it, name, end - start);
}

typedef void (*BLEND_SOA_FUNC)(PixelFloatSoA *dst, PixelFloatSoA *src, int pixel_count,
                               uint16_t opacity, int blend_mode);
void bench_load_soa(char *name, int it, BLEND_SOA_FUNC func)
{
    // random pixels
    PixelFloatSoA src = {
        .b = DP_malloc(DP_TILE_LENGTH * sizeof(float)),
        .g = DP_malloc(DP_TILE_LENGTH * sizeof(float)),
        .r = DP_malloc(DP_TILE_LENGTH * sizeof(float)),
        .a = DP_malloc(DP_TILE_LENGTH * sizeof(float)),
    };
    PixelFloatSoA dst = {
        .b = DP_malloc(DP_TILE_LENGTH * sizeof(float)),
        .g = DP_malloc(DP_TILE_LENGTH * sizeof(float)),
        .r = DP_malloc(DP_TILE_LENGTH * sizeof(float)),
        .a = DP_malloc(DP_TILE_LENGTH * sizeof(float)),
    };

    double start = clock_ms();
    for (int i = 0; i < it; i++) {
        func(&dst, &src, DP_TILE_LENGTH, DP_BIT15, DP_BLEND_MODE_NORMAL);
    }
    double end = clock_ms();

    printf("it: %d %s: %fms\n", it, name, end - start);
}
DISABLE_OPT_END


int main_()
{
    int width, height;

    DP_Pixel15 *src = load_image("src.png", &width, &height);
    DP_Pixel15 *dst = load_image("dst.png", &width, &height);

    int count = width * height;

    DP_Pixel15 *dst2 = DP_malloc(count * sizeof(DP_Pixel15));
    memcpy(dst2, dst, count * sizeof(DP_Pixel15));

    PixelFloatSoA src_float = {
        .b = DP_malloc(count * sizeof(float)),
        .g = DP_malloc(count * sizeof(float)),
        .r = DP_malloc(count * sizeof(float)),
        .a = DP_malloc(count * sizeof(float)),
    };
    to_p(src, &src_float, count);

    PixelFloatSoA dst_float = {
        .b = DP_malloc(count * sizeof(float)),
        .g = DP_malloc(count * sizeof(float)),
        .r = DP_malloc(count * sizeof(float)),
        .a = DP_malloc(count * sizeof(float)),
    };
    to_p(dst, &dst_float, count);


    for (int i = 0; i < 1; i++) {
        DP_blend_pixels(dst, src, count, DP_BIT15, DP_BLEND_MODE_NORMAL);
        DP_blend_pixels_simd(dst2, src, count, DP_BIT15, DP_BLEND_MODE_NORMAL);
        DP_blend_pixels_simd_float_soa(&dst_float, &src_float, count, DP_BIT15, DP_BLEND_MODE_NORMAL);
    }

    DP_Pixel15 *p_dst = DP_malloc(count * sizeof(DP_Pixel15));
    from_p(&dst_float, p_dst, count);

    save_image("seq.png", dst, width, height);
    save_image("simd.png", dst2, width, height);
    save_image("simd_float.png", p_dst, width, height);

    DP_free(src);
    DP_free(dst);
    DP_free(dst2);
}

int main()
{

    int it = 500000;
    bench_load("clang autovec", it, DP_blend_pixels);
    bench_load("simd intrinsics", it, DP_blend_pixels_simd);
    bench_load_soa("simd float SoA", it, DP_blend_pixels_simd_float_soa);
}
