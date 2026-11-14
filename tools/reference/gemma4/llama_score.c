/* Teacher-forced scoring with llama.cpp (libllama), the reference lily's
 * `lily-gemma4 score` is compared with: same GGUF, same token ids.
 *
 * Cases file: one case a line, `id<TAB>first<TAB>t0,t1,...` (token ids,
 * BOS included); tokens first.. are scored, each from the logits of the
 * position before it. Prints per case: scored tokens, mean NLL, greedy
 * agreement; with --dump F every scored position's log-softmax as f16
 * (n_vocab values) goes to F; with --base F those of another model are read
 * back and the KL divergence base -> this model and the top-1 agreement
 * with the base are printed too. CPU only (no device offload), mmap.
 *
 *   cc -O2 llama_score.c -lllama -lggml-base -lm -o llama_score
 *   llama_score model.gguf cases.tsv [--dump F | --base F] [--threads N]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "llama.h"
#include "ggml.h"

static int argmax(const float *x, int n) {
    int b = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[b]) b = i;
    return b;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.gguf cases.tsv [--dump F|--base F] [--threads N]\n", argv[0]);
        return 2;
    }
    const char *dump = NULL, *base = NULL;
    int threads = 16;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--dump") && i + 1 < argc) dump = argv[++i];
        else if (!strcmp(argv[i], "--base") && i + 1 < argc) base = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
    }
    FILE *in = fopen(argv[2], "r");
    if (!in) { perror(argv[2]); return 1; }
    /* Read the cases. */
    size_t cap = 1 << 20;
    char *line = malloc(cap);
    int ncase = 0, maxlen = 0;
    char **ids = NULL; int *firsts = NULL; int **toks = NULL; int *lens = NULL;
    while (fgets(line, (int)cap, in)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        char *id = strtok(line, "\t");
        char *first = strtok(NULL, "\t");
        char *list = strtok(NULL, "\t\n");
        if (!id || !first || !list) continue;
        ids = realloc(ids, sizeof *ids * (ncase + 1));
        firsts = realloc(firsts, sizeof *firsts * (ncase + 1));
        toks = realloc(toks, sizeof *toks * (ncase + 1));
        lens = realloc(lens, sizeof *lens * (ncase + 1));
        ids[ncase] = strdup(id);
        firsts[ncase] = atoi(first);
        int n = 1;
        for (char *c = list; *c; c++) n += *c == ',';
        toks[ncase] = malloc(sizeof(int) * n);
        n = 0;
        for (char *t = strtok(list, ","); t; t = strtok(NULL, ",")) toks[ncase][n++] = atoi(t);
        lens[ncase] = n;
        if (n > maxlen) maxlen = n;
        ncase++;
    }
    fclose(in);

    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t none[1] = { NULL };
    mp.devices = none;
    mp.n_gpu_layers = 0;
    struct llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "cannot load %s\n", argv[1]); return 1; }
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    const int nv = llama_vocab_n_tokens(vocab);
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t)(maxlen + 16);
    cp.n_batch = cp.n_ubatch = 512;
    cp.n_threads = cp.n_threads_batch = threads;
    cp.op_offload = false;
    struct llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "cannot create a context\n"); return 1; }

    FILE *df = dump ? fopen(dump, "wb") : NULL;
    FILE *bf = base ? fopen(base, "rb") : NULL;
    if ((dump && !df) || (base && !bf)) { perror(dump ? dump : base); return 1; }
    ggml_fp16_t *h = malloc(sizeof(ggml_fp16_t) * nv);
    float *lp = malloc(sizeof(float) * nv);
    double tot_nll = 0, tot_kld = 0;
    long tot_n = 0, tot_agree = 0, tot_same = 0;
    printf("case\tscored\tnll\tagree\tkld\tsame_top\n");
    struct llama_batch batch = llama_batch_init(512, 0, 1);
    for (int c = 0; c < ncase; c++) {
        llama_memory_clear(llama_get_memory(ctx), true);
        const int n = lens[c], first = firsts[c];
        double nll = 0, kld = 0;
        int agree = 0, same = 0, scored = 0;
        for (int b0 = 0; b0 < n - 1; b0 += 512) {
            const int bn = (n - 1 - b0) < 512 ? (n - 1 - b0) : 512;
            batch.n_tokens = bn;
            for (int i = 0; i < bn; i++) {
                const int p = b0 + i;
                batch.token[i] = toks[c][p];
                batch.pos[i] = p;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = p + 1 >= first;
            }
            if (llama_decode(ctx, batch)) { fprintf(stderr, "decode failed\n"); return 1; }
            for (int i = 0; i < bn; i++) {
                const int p = b0 + i;
                if (p + 1 < first) continue;
                const float *l = llama_get_logits_ith(ctx, i);
                float mx = l[0];
                for (int v = 1; v < nv; v++) if (l[v] > mx) mx = l[v];
                double s = 0;
                for (int v = 0; v < nv; v++) s += exp((double)(l[v] - mx));
                const double ls = log(s);
                for (int v = 0; v < nv; v++) lp[v] = (float)((double)(l[v] - mx) - ls);
                const int target = toks[c][p + 1];
                nll -= lp[target];
                const int top = argmax(lp, nv);
                agree += top == target;
                if (df) {
                    for (int v = 0; v < nv; v++) h[v] = ggml_fp32_to_fp16(lp[v]);
                    fwrite(h, sizeof *h, nv, df);
                }
                if (bf) {
                    if (fread(h, sizeof *h, nv, bf) != (size_t)nv) { fprintf(stderr, "base ends early\n"); return 1; }
                    double k = 0;
                    int btop = 0;
                    float bmax = -INFINITY;
                    for (int v = 0; v < nv; v++) {
                        const float b = ggml_fp16_to_fp32(h[v]);
                        if (b > bmax) { bmax = b; btop = v; }
                        if (b > -60.f) k += exp((double)b) * ((double)b - lp[v]);
                    }
                    kld += k;
                    same += btop == top;
                }
                scored++;
            }
        }
        printf("%s\t%d\t%.6f\t%.2f%%\t%.6f\t%.2f%%\n", ids[c], scored, nll / scored,
               100.0 * agree / scored, kld / scored, 100.0 * same / scored);
        fflush(stdout);
        tot_nll += nll; tot_kld += kld; tot_n += scored; tot_agree += agree; tot_same += same;
    }
    printf("total\t%ld\t%.6f\t%.2f%%\t%.6f\t%.2f%%\n", tot_n, tot_nll / tot_n,
           100.0 * tot_agree / tot_n, tot_kld / tot_n, 100.0 * tot_same / tot_n);
    if (df) fclose(df);
    if (bf) fclose(bf);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
