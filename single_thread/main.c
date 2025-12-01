#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/qap.h"

typedef struct
{
    double T0;
    double alpha;
    size_t iters;
    unsigned long long seed;
    int random_start;
    const char *log_path;
} SAParams;

typedef struct
{
    double best_cost;
    size_t best_iter;
    double final_cost;
} SAStats;

void sa_single_run(
    const QAPProblem *P,
    const SAParams *cfg,
    SAStats *out_stats,
    FILE *log);

static void usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s --input <path> [options]\n"
        "Options:\n"
        "  --iters N     number of iterations\n"
        "  --T0 X        start temperature\n"
        "  --alpha A     cooling factor\n"
        "  --seed S      RNG seed\n"
        "  --random      random start permutation\n"
        "  --log FILE    write log to FILE\n",
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
        .random_start = 0,
        .log_path = NULL
    };

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
        else if (!strcmp(argv[i], "--log") && i + 1 < argc)
        {
            cfg.log_path = argv[++i];
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

    FILE *log = NULL;
    if (cfg.log_path)
    {
        log = fopen(cfg.log_path, "w");
        if (!log)
        {
            perror("fopen(log)");
            qap_free(P);
            return 3;
        }
    }

    SAStats st = {0};
    sa_single_run(P, &cfg, &st, log);

    if (log)
        fclose(log);

    qap_free(P);
    return 0;
}
