/* Golden data for lily's DeepSeek V4 Flash CPU reference: inputs and the
 * outputs of ds4's own reference functions (ds4.c, antirez, MIT): the FP8 KV
 * round trip, the indexer's Hadamard + FP4 simulation, the E4M3/E2M1
 * roundings, RoPE with YaRN (rope_tail_ext_inplace with the parameters of
 * rope_tail_layer_inplace) and the compressor's softmax pool
 * (compressor_pool_decode_state).
 *
 * Build and run against a ds4 checkout (b3f5018), from a scratch directory:
 *   sed -n 3722,3935p $DS4/ds4.c > qat.inc
 *   sed -n 12480,12548p $DS4/ds4.c > rope.inc
 *   sed -n 14723,14775p $DS4/ds4.c > pool.inc
 *   cc -O1 -ffp-contract=off -I. ops_golden.c -lm -o gen
 *   ./gen tests/goldens/deepseek4_ops.bin
 *
 * File: a sequence of records `u32 n, f32[n]`, in the order written below
 * (each case's inputs, then its outputs). */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define DS4_NEG_INF (-1.0e30f)
static void ds4_die(const char *m) { fprintf(stderr, "%s\n", m); exit(1); }
#include "qat.inc"
#include "rope.inc"
#include "pool.inc"

static FILE *out;
static void rec(const float *x, uint32_t n) {
    fwrite(&n, 4, 1, out);
    fwrite(x, 4, n, out);
}
static uint64_t rng = 0x9e3779b97f4a7c15ull;
static float frand(void) { /* uniform in [-1, 1) */
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (float)((rng >> 40) & 0xffffff) / 8388608.0f - 1.0f;
}
/* Values over several decades, some exact ties of the formats. */
static void fill(float *x, uint32_t n, float mag) {
    for (uint32_t i = 0; i < n; i++) {
        float v = frand() * mag * powf(10.0f, 2.0f * frand());
        if (i % 37 == 5) v = 0.0f;
        if (i % 41 == 7) v = mag * 1.0e-7f * frand();
        x[i] = v;
    }
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    out = fopen(argv[1], "wb");
    float x[4096], y[4096];

    /* 1. FP8 KV round trip: 6 rows of 512 (n_rot 64) at several scales. */
    const float mags[6] = {1e-6f, 1e-3f, 0.1f, 1.0f, 30.0f, 3000.0f};
    for (int r = 0; r < 6; r++) {
        fill(x, 512, mags[r]);
        rec(x, 512);
        memcpy(y, x, sizeof(float) * 512);
        dsv4_fp8_kv_quantize_row_inplace_cpu(y, 512, 64);
        rec(y, 512);
    }
    /* 2. Indexer Hadamard + FP4: 6 rows of 128. */
    for (int r = 0; r < 6; r++) {
        fill(x, 128, mags[r]);
        rec(x, 128);
        memcpy(y, x, sizeof(float) * 128);
        dsv4_indexer_qat_row_inplace_cpu(y, 128);
        rec(y, 128);
    }
    /* 3. E4M3 and E2M1 roundings of a sweep (incl. ties and the clamps). */
    for (int i = 0; i < 2048; i++) x[i] = (float)(i - 1024) * 0.25f + (i % 3) * 0.0009765625f;
    rec(x, 2048);
    for (int i = 0; i < 2048; i++) y[i] = dsv4_e4m3fn_dequant_cpu(x[i]);
    rec(y, 2048);
    for (int i = 0; i < 2048; i++) x[i] = (float)(i - 1024) / 128.0f;
    rec(x, 2048);
    for (int i = 0; i < 2048; i++) y[i] = dsv4_e2m1fn_dequant_cpu(x[i]);
    rec(y, 2048);

    /* 4. RoPE: plain (base 1e4) and compressed (base 160000, YaRN factor 16,
     * original context 65536, beta 32/1, magnitude cancelled), forward and
     * inverse, two heads of 512 and one of 128, at several positions. */
    const uint32_t pos[8] = {0, 1, 7, 127, 128, 1000, 65535, 300001};
    for (int yarn = 0; yarn < 2; yarn++) {
        const float base = yarn ? 160000.0f : 10000.0f;
        const float fs = yarn ? 1.0f / 16.0f : 1.0f;
        const float ext = yarn ? 1.0f : 0.0f;
        float attn = 1.0f;
        if (yarn) attn /= 1.0f + 0.1f * logf(1.0f / fs);
        for (int p = 0; p < 8; p++) {
            for (int inv = 0; inv < 2; inv++) {
                for (int hd = 0; hd < 2; hd++) {
                    const uint32_t head_dim = hd ? 128 : 512;
                    const uint32_t n_head = hd ? 1 : 2;
                    fill(x, head_dim * n_head, 1.0f);
                    rec(x, head_dim * n_head);
                    memcpy(y, x, sizeof(float) * head_dim * n_head);
                    rope_tail_ext_inplace(y, n_head, head_dim, 64, pos[p],
                                          yarn ? 65536 : 0, base, fs, ext, attn,
                                          32.0f, 1.0f, inv != 0);
                    rec(y, head_dim * n_head);
                }
            }
        }
    }

    /* 5. Compressor pool: ratio 4 (two lanes) and ratio 128, head_dim 16;
     * some rows empty (-inf scores), scores over a few units. */
    const uint32_t ratios[2] = {4, 128};
    for (int k = 0; k < 2; k++) {
        const uint32_t r = ratios[k], hd = 16, coff = r == 4 ? 2 : 1;
        const uint32_t n = coff * r * coff * hd;
        float kv[2 * 128 * 2 * 16], sc[2 * 128 * 2 * 16];
        for (int c = 0; c < 2; c++) {
            fill(kv, n, 1.0f);
            for (uint32_t i = 0; i < n; i++) sc[i] = 4.0f * frand();
            if (c == 0) /* the first window: rows 0..r-1 never written */
                for (uint32_t i = 0; i < (r == 4 ? r * coff * hd : 0); i++) sc[i] = DS4_NEG_INF;
            rec(kv, n);
            rec(sc, n);
            compressor_pool_decode_state(y, kv, sc, hd, r);
            rec(y, hd);
        }
    }
    fclose(out);
    return 0;
}
