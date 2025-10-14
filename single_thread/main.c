#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../common/qap.h"

// Defined in sa_single.c
typedef struct
{
    double T0;
    double alpha;
    size_t iters;
    unsigned long long seed;
    int random_start;
} SAParams;

typedef struct
{
    double best_cost;
    size_t best_iter;
    double final_cost;
} SAStats;

void sa_single_run(const QAPProblem *P, const SAParams *cfg, SAStats *out_stats);

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s --input <path> [--iters N] [--T0 X] [--alpha A] [--seed S] [--random]\n"
            "  --input   path to QAP instance (txt)\n"
            "  --iters   number of SA iterations (default: 100000)\n"
            "  --T0      initial temperature (default: 1e4)\n"
            "  --alpha   cooling factor in (0,1) (default: 0.9995)\n"
            "  --seed    RNG seed (default: 0 -> time-based)\n"
            "  --random  start from a random permutation (default: identity)\n",
            prog);
}

int main(int argc, char **argv)
{
    const char *input = NULL;
    SAParams cfg = {
        .T0 = 1e4,
        .alpha = 0.9995,
        .iters = 100000,
        .seed = 0ULL,
        .random_start = 0};

    // argv parser.
    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "--input") && i + 1 < argc)
        {
            input = argv[++i];
        }
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc)
        {
            cfg.iters = (size_t)strtoull(argv[++i], NULL, 10);
        }
        else if (!strcmp(argv[i], "--T0") && i + 1 < argc)
        {
            cfg.T0 = strtod(argv[++i], NULL);
        }
        else if (!strcmp(argv[i], "--alpha") && i + 1 < argc)
        {
            cfg.alpha = strtod(argv[++i], NULL);
        }
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc)
        {
            cfg.seed = strtoull(argv[++i], NULL, 10);
        }
        else if (!strcmp(argv[i], "--random"))
        {
            cfg.random_start = 1;
        }
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
        {
            usage(argv[0]);
            return 0;
        }
        else
        {
            fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (!input)
    {
        fprintf(stderr, "Error: --input is required\n");
        usage(argv[0]);
        return 1;
    }

    QAPProblem *P = qap_load_from_file(input);
    if (!P)
    {
        fprintf(stderr, "Failed to load QAP instance: %s\n", input);
        return 2;
    }

    // Main running.
    SAStats st = {0};
    sa_single_run(P, &cfg, &st);

    qap_free(P);
    return 0;
}
