/* Golden dequantization data for the GGUF block types Gemma 4's files add to
 * lily (tests/goldens/gguf_dequant_gemma4.bin): records of (type id, block
 * count, raw blocks, decoded f32 values) in the format of
 * tests/goldens/gguf_dequant.bin. The blocks are real ones, read from the
 * GGUF files given (a few runs from several tensors of each wanted type);
 * the values are ggml's own dequantization (`to_float` of the type's
 * traits, the library llama.cpp runs).
 *
 * Build and run against an installed ggml (llama.cpp 0.6.0 / ggml 0.26):
 *   cc -O1 gguf_dequant_golden.c -lggml-base -o gen
 *   ./gen out.bin file.gguf...
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ggml.h"
#include "gguf.h"

/* The types wanted, and how many records of each so far. */
static const enum ggml_type WANT[] = {
    GGML_TYPE_Q4_0, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ4_NL,
    GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,
};
#define NWANT (sizeof WANT / sizeof WANT[0])
#define PER_TYPE 6   /* tensors sampled per type */
#define BLOCKS 16    /* consecutive blocks per sample */

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s out.bin file.gguf...\n", argv[0]); return 2; }
    FILE *out = fopen(argv[1], "wb");
    if (!out) { perror(argv[1]); return 1; }
    int count[NWANT] = {0};
    for (int f = 2; f < argc; f++) {
        struct gguf_init_params p = { .no_alloc = true, .ctx = NULL };
        struct gguf_context *g = gguf_init_from_file(argv[f], p);
        if (!g) { fprintf(stderr, "cannot read %s\n", argv[f]); return 1; }
        FILE *in = fopen(argv[f], "rb");
        const size_t data = gguf_get_data_offset(g);
        const int64_t n = gguf_get_n_tensors(g);
        for (int64_t t = 0; t < n; t++) {
            const enum ggml_type ty = gguf_get_tensor_type(g, t);
            int w = -1;
            for (size_t i = 0; i < NWANT; i++) if (WANT[i] == ty) w = (int)i;
            if (w < 0 || count[w] >= PER_TYPE) continue;
            const size_t bb = ggml_type_size(ty);
            const int64_t bv = ggml_blck_size(ty);
            const size_t size = gguf_get_tensor_size(g, t);
            /* A run from the middle of the tensor. */
            const size_t nblk = size / bb;
            const size_t first = (nblk / 2) / BLOCKS * BLOCKS;
            uint8_t *raw = malloc(BLOCKS * bb);
            float *vals = malloc(sizeof(float) * BLOCKS * bv);
            if (fseeko(in, (off_t)(data + gguf_get_tensor_offset(g, t) + first * bb), SEEK_SET) ||
                fread(raw, bb, BLOCKS, in) != BLOCKS) {
                free(raw); free(vals);
                continue; /* past a partial download */
            }
            ggml_get_type_traits(ty)->to_float(raw, vals, BLOCKS * bv);
            const uint32_t hdr[2] = { (uint32_t)ty, BLOCKS };
            fwrite(hdr, 4, 2, out);
            fwrite(raw, bb, BLOCKS, out);
            fwrite(vals, sizeof(float), BLOCKS * bv, out);
            fprintf(stderr, "%s: %s %s blocks %zu..\n", argv[f], ggml_type_name(ty),
                    gguf_get_tensor_name(g, t), first);
            count[w]++;
            free(raw); free(vals);
        }
        fclose(in);
        gguf_free(g);
    }
    fclose(out);
    for (size_t i = 0; i < NWANT; i++) fprintf(stderr, "%s: %d records\n", ggml_type_name(WANT[i]), count[i]);
    return 0;
}
