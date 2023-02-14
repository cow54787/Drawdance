/*
 * Copyright (c) 2022
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef DPCOMMON_CPU_H
#define DPCOMMON_CPU_H
#include "common.h"

#if defined(_M_X64) || defined(__x86_64__)
#    define DP_CPU_X64
#endif

#define DP_DO_PRAGMA_(x) _Pragma(#x)
#define DP_DO_PRAGMA(x)  DP_DO_PRAGMA_(x)

#if defined(__clang__)
#    define DP_TARGET_BEGIN(TARGET) DP_DO_PRAGMA(clang attribute push(__attribute__((target(TARGET))), apply_to = function))
#    define DP_TARGET_END           _Pragma("clang attribute pop")
#elif defined(__GNUC__)
#    define DP_TARGET_BEGIN(TARGET) _Pragma("GCC push_options") DP_DO_PRAGMA(GCC target(TARGET))
#    define DP_TARGET_END           _Pragma("GCC pop_options");
#else
#    define DP_TARGET_BEGIN(TARGET) // nothing
#    define DP_TARGET_END           // nothing
#endif

// Order matters
typedef enum DP_CPU_SUPPORT {
    DP_CPU_SUPPORT_DEFAULT,
#ifdef DP_CPU_X64
    DP_CPU_SUPPORT_SSE42,
    DP_CPU_SUPPORT_AVX2,
#endif
} DP_CPU_SUPPORT;


DP_CPU_SUPPORT DP_get_cpu_support(void);

#endif
