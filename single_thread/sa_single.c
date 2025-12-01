#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "../common/qap.h"

#define BOTTOM_TEMPERATURE_BOUND 1e-6

// ---------------- RNG ----------------

typedef struct
{
    unsigned long long s;
} rng_t;

static inline void rng_seed(rng_t *r, unsigned long long seed)
{
    r->s = seed ? seed : 0x9E3779B97F4A7C15ULL;
}

static inline unsigned long long rng_u64(rng_t *r)
{
    unsigned long long x = r->s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    r->s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static inline double rng_uniform(rng_t *r)
{
    const unsigned long long m =
        0x3FF0000000000000ULL | (rng_u64(r) >> 12);
    double d;
    memcpy(&d, &m, sizeof(d));
    return d - 1.0;
}

static inline size_t rng_randint(rng_t *r, size_t n)
{
    return (size_t)(rng_u64(r) % (n ? n : 1));
}

// ---------------- Solution helpers ----------------

static void sol_alloc(QAPSolution *s, size_t n)
{
    s->n = n;
    s->permutations = (size_t *)malloc(n * sizeof(size_t));
    s->cost = 0.0;

    if (!s->permutations)
    {
        fprintf(stderr, "alloc perm failed\n");
        exit(1);
    }
}

static void sol_free(QAPSolution *s)
{
    free(s->permutations);
    s->permutations = NULL;
    s->n = 0;
    s->cost = 0.0;
}

static void sol_identity(QAPSolution *s)
{
    for (size_t i = 0; i < s->n; i++)
        s->permutations[i] = i;
}

static void sol_shuffle(QAPSolution *s, rng_t *r)
{
    for (size_t i = s->n - (s->n > 0); i > 0; --i)
    {
        size_t j = rng_randint(r, i + 1);
        size_t t = s->permutations[i];
        s->permutations[i] = s->permutations[j];
        s->permutations[j] = t;
    }
}

static void sol_copy(const QAPSolution *src, QAPSolution *dst)
{
    if (dst->n != src->n)
    {
        free(dst->permutations);
        sol_alloc(dst, src->n);
    }

    memcpy(dst->permutations,
           src->permutations,
           src->n * sizeof(size_t));

    dst->cost = src->cost;
}

// ---------------- SA ----------------

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
    FILE *log)
{
    rng_t rng;
    rng_seed(&rng,
             cfg->seed ? cfg->seed : (unsigned long long)time(NULL));

    QAPSolution current;
    sol_alloc(&current, P->n);

    QAPSolution best;
    sol_alloc(&best, P->n);

    if (cfg->random_start)
    {
        sol_identity(&current);
        sol_shuffle(&current, &rng);
    }
    else
    {
        sol_identity(&current);
    }

    current.cost = qap_eval(P, &current);
    sol_copy(&current, &best);

    double T = cfg->T0;
    const double alpha = cfg->alpha;

    if (log)
    {
        fprintf(log, "# iter current_cost best_cost T\n");
        fprintf(log, "0 %.10f %.10f %.10f\n",
                current.cost, best.cost, T);
    }

    clock_t start = clock();

    for (size_t it = 0; it < cfg->iters; it++)
    {
        size_t i = rng_randint(&rng, current.n);
        size_t j = rng_randint(&rng, current.n);
        if (i == j)
            continue;

        double delta = qap_delta_swap(P, &current, i, j);

        int accept = 0;
        if (delta < 0.0)
        {
            accept = 1;
        }
        else
        {
            double u = rng_uniform(&rng);
            double prob = (T > 0.0) ? exp(-delta / T) : 0.0;
            if (u < prob)
                accept = 1;
        }

        if (accept)
        {
            qap_apply_swap(&current, i, j, delta);

            if (current.cost < best.cost)
            {
                sol_copy(&current, &best);
                if (out_stats)
                {
                    out_stats->best_cost = best.cost;
                    out_stats->best_iter = it + 1;
                }
            }
        }

        T *= alpha;
        if (T < BOTTOM_TEMPERATURE_BOUND)
            T = BOTTOM_TEMPERATURE_BOUND;

        if (log && (it % 10 == 0))
        {
            fprintf(log, "%zu %.10f %.10f %.10f\n",
                    it, current.cost, best.cost, T);
        }
    }

    clock_t end = clock();

    if (out_stats)
        out_stats->final_cost = current.cost;

    double elapsed_time =
        (double)(end - start) / CLOCKS_PER_SEC;

    if (log)
    {
        fprintf(log,
                "\n# FINAL\n%.10f\n# BEST\n%.10f\n# TIME\n%f\n",
                current.cost, best.cost, elapsed_time);
    }

    printf("FINAL: %.10f  BEST: %.10f  BEST_ITER: %zu\n",
           current.cost, best.cost,
           (out_stats ? out_stats->best_iter : 0));

    printf("Elapsed time: %f\n", elapsed_time);

    sol_free(&current);
    sol_free(&best);
}
