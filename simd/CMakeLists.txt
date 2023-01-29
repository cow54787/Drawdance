# Copyright (c) 2022 askmeaboutloom
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

set(simd_sources main.c)

add_clang_format_files("${simd_sources}")

add_executable(simd "${simd_sources}")
set_dp_target_properties(simd)
target_link_libraries(simd PUBLIC dpcommon dpengine dpmsg)

find_package(benchmark CONFIG REQUIRED)

set(simd_bench_sources simd_bench.cpp)
add_clang_format_files("${simd_bench_sources}")

add_executable(simd_bench "${simd_bench_sources}")
set_dp_target_properties(simd_bench CXX NO_WARNINGS)
target_link_libraries(simd_bench PUBLIC dpcommon dpengine dpmsg benchmark::benchmark benchmark::benchmark_main)
