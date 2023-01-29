void __fastcall DP_blend_pixels_simd(
    DP_Pixel15_0 *dst,
    DP_Pixel15_0 *src,
    int pixel_count,
    uint16_t opacity,
    int blend_mode)
{
    __m128i v5;          // xmm10
    unsigned __int64 v6; // rcx
    __m128i si128;       // xmm8
    __m128i v8;          // xmm5
    __m128i v9;          // xmm0
    __m128i v10;         // xmm3
    __m128i v11;         // xmm5
    __m128i v12;         // xmm4
    __m128i v13;         // xmm7
    __m128i v14;         // xmm1
    __m128i v15;         // xmm0
    __m128i v16;         // xmm7
    __m128i v17;         // xmm6
    __m128i v18;         // xmm2
    __m128i v19;         // xmm3
    __m128i v20;         // xmm4
    __m128i v21;         // xmm0
    __m128i v22;         // xmm3

    if (pixel_count) {
        v5 = _mm_shuffle_epi32(_mm_cvtsi32_si128(opacity), 0);
        v6 = 0LL;
        si128 = _mm_load_si128((const __m128i *)&xmmword_2010);
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
                    _mm_srli_epi32(_mm_mullo_epi32(_mm_blend_epi16(v10, (__m128i)0LL, 170), v5), 0xFu),
                    _mm_srli_epi32(_mm_mullo_epi32(_mm_blend_epi16(v15, (__m128i)0LL, 170), v18), 0xFu)),
                _mm_add_epi32(
                    _mm_srli_epi32(_mm_mullo_epi32(_mm_blend_epi16(v11, (__m128i)0LL, 170), v5), 0xFu),
                    _mm_srli_epi32(_mm_mullo_epi32(_mm_blend_epi16(v16, (__m128i)0LL, 170), v18), 0xFu)));
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
