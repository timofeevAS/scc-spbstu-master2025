#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "sprng.h"

#define IDX(n,i,j) ((i) * (n) + (j))

#define BOTTOM_TEMPERATURE_BOUND 1e-6
#define LOG_EVERY_ITER 1000
#define SEED 12345

// ===================== QAP structures =====================

typedef struct
{
    size_t n;
    double *flow;      // n x n
    double *distance;  // n x n
} QAPProblem;

typedef struct
{
    size_t n;
    size_t *permutations;
    double cost;
} QAPSolution;


// ===================== QAP load / free =====================

QAPProblem *qap_load_from_file(const char *path)
{
    FILE *fp = fopen(path, "r");
    if (!fp)
    {
        fprintf(stderr, "Cannot open file: %s\n", path);
        return NULL;
    }

    size_t n;
    if (fscanf(fp, "%zu", &n) != 1)
    {
        fprintf(stderr, "Failed to read n from %s\n", path);
        fclose(fp);
        return NULL;
    }

    double *flow = malloc(n * n * sizeof(double));
    double *dist = malloc(n * n * sizeof(double));

    if (!flow || !dist)
    {
        fprintf(stderr, "Memory alloc failed for matrices\n");
        free(flow);
        free(dist);
        fclose(fp);
        return NULL;
    }

    for (size_t i = 0; i < n * n; i++)
    {
        if (fscanf(fp, "%lf", &flow[i]) != 1)
        {
            fprintf(stderr, "Failed to read flow[%zu]\n", i);
            free(flow);
            free(dist);
            fclose(fp);
            return NULL;
        }
    }

    for (size_t i = 0; i < n * n; i++)
    {
        if (fscanf(fp, "%lf", &dist[i]) != 1)
        {
            fprintf(stderr, "Failed to read dist[%zu]\n", i);
            free(flow);
            free(dist);
            fclose(fp);
            return NULL;
        }
    }

    fclose(fp);

    QAPProblem *p = malloc(sizeof(QAPProblem));
    if (!p)
    {
        fprintf(stderr, "Memory alloc failed for QAPProblem\n");
        free(flow);
        free(dist);
        return NULL;
    }

    p->n = n;
    p->flow = flow;
    p->distance = dist;
    return p;
}

void qap_free(QAPProblem *p)
{
    if (!p) return;
    free(p->flow);
    free(p->distance);
    free(p);
}


// ===================== QAP eval / delta / apply =====================

double qap_eval(const QAPProblem *p, const QAPSolution *s)
{
    double cost = 0.0;
    size_t n = p->n;

    for (size_t i = 0; i < n; i++)
    {
        size_t li = s->permutations[i];
        for (size_t j = 0; j < n; j++)
        {
            size_t lj = s->permutations[j];
            cost += p->flow[IDX(n,i,j)] * p->distance[IDX(n,li,lj)];
        }
    }
    return cost;
}

double qap_delta_swap(const QAPProblem *p, const QAPSolution *s, size_t i, size_t j)
{
    if (i == j) return 0.0;

    if (i > j)
    {
        size_t t = i;
        i = j;
        j = t;
    }

    size_t n = p->n;
    const double *F = p->flow;
    const double *D = p->distance;
    const size_t *perm = s->permutations;

    size_t pi = perm[i];
    size_t pj = perm[j];

    double delta = 0.0;

    for (size_t k = 0; k < n; k++)
    {
        if (k == i || k == j) continue;

        size_t pk = perm[k];

        delta += (F[IDX(n,i,k)] - F[IDX(n,j,k)]) * (D[IDX(n,pj,pk)] - D[IDX(n,pi,pk)])
               + (F[IDX(n,k,i)] - F[IDX(n,k,j)]) * (D[IDX(n,pk,pj)] - D[IDX(n,pk,pi)]);
    }

    delta += (F[IDX(n,i,i)] - F[IDX(n,j,j)]) * (D[IDX(n,pj,pj)] - D[IDX(n,pi,pi)])
           + (F[IDX(n,i,j)] - F[IDX(n,j,i)]) * (D[IDX(n,pj,pi)] - D[IDX(n,pi,pj)]);

    return delta;
}

void qap_apply_swap(QAPSolution *s, size_t i, size_t j, double delta)
{
    size_t tmp = s->permutations[i];
    s->permutations[i] = s->permutations[j];
    s->permutations[j] = tmp;
    s->cost += delta;
}


// ===================== Solution helpers =====================

QAPSolution *qap_solution_alloc(size_t n)
{
    QAPSolution *s = malloc(sizeof(QAPSolution));
    if (!s)
    {
        fprintf(stderr, "Alloc QAPSolution failed\n");
        exit(1);
    }
    s->n = n;
    s->permutations = malloc(n * sizeof(size_t));
    if (!s->permutations)
    {
        fprintf(stderr, "Alloc permutations failed\n");
        free(s);
        exit(1);
    }
    s->cost = 0.0;
    return s;
}

void qap_solution_free(QAPSolution *s)
{
    if (!s) return;
    free(s->permutations);
    free(s);
}

void qap_solution_copy(QAPSolution *dst, const QAPSolution *src)
{
    dst->n = src->n;
    dst->cost = src->cost;
    memcpy(dst->permutations, src->permutations, src->n * sizeof(size_t));
}

static void sol_identity(QAPSolution *s)
{
    for (size_t i = 0; i < s->n; i++)
        s->permutations[i] = i;
}

static void sol_shuffle(QAPSolution *s, int *stream)
{
    for (size_t i = s->n - 1; i > 0; i--)
    {
        size_t j = (size_t)(sprng(stream) * (i + 1));
        size_t t = s->permutations[i];
        s->permutations[i] = s->permutations[j];
        s->permutations[j] = t;
    }
}


// ===================== RNG wrappers =====================

static inline size_t rng_randint(int *stream, size_t n)
{
    return (size_t)(sprng(stream) * n);
}

static inline double rng_uniform(int *stream)
{
    return sprng(stream);
}


// ===================== GLOBAL BEST =====================

static QAPSolution *g_best = NULL;


// ===================== MAIN (OpenMP SA) =====================

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s instance.dat [iters]\n", argv[0]);
        return 1;
    }

    const char *path = argv[1];
    long long iters = (argc > 2 ? atoll(argv[2]) : 100000);

    QAPProblem *P = qap_load_from_file(path);
    if (!P)
    {
        fprintf(stderr, "Failed to load QAP instance\n");
        return 1;
    }

    size_t n = P->n;
    int nthreads = omp_get_max_threads();

    printf("OpenMP threads = %d\n", nthreads);
    printf("iters          = %lld\n", iters);
    printf("random_start   = 1 (always)\n");

    // global best
    g_best = qap_solution_alloc(n);
    g_best->cost = HUGE_VAL;

    // =============== PARALLEL SA REGION ===============

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        // SPRNG stream per thread
        int *stream = init_sprng(
            DEFAULT_RNG_TYPE,
            tid,
            nthreads,
            SEED,
            SPRNG_DEFAULT
        );

        QAPSolution *cur  = qap_solution_alloc(n);
        QAPSolution *best = qap_solution_alloc(n);

        // random_start = 1 => identity + shuffle
        sol_identity(cur);
        sol_shuffle(cur, stream);

        cur->cost = qap_eval(P, cur);
        qap_solution_copy(best, cur);

        double T = 300000.0;
        double alpha = 0.9992;

        long long max_logs = iters / LOG_EVERY_ITER + 5;
        double *trace = malloc(max_logs * sizeof(double));
        double *timer = malloc(max_logs * sizeof(double));
        long long tlen = 0;

        double t_start = omp_get_wtime();

        for (long long it = 0; it < iters; it++)
        {
            size_t i = rng_randint(stream, n);
            size_t j = rng_randint(stream, n);
            if (i == j) continue;

            double delta = qap_delta_swap(P, cur, i, j);

            int accept = 0;
            if (delta < 0.0)
                accept = 1;
            else if (rng_uniform(stream) < exp(-delta / T))
                accept = 1;

            if (accept)
            {
                qap_apply_swap(cur, i, j, delta);
                if (cur->cost < best->cost)
                    qap_solution_copy(best, cur);
            }

            T *= alpha;
            if (T < BOTTOM_TEMPERATURE_BOUND)
                T = BOTTOM_TEMPERATURE_BOUND;

            if (it % LOG_EVERY_ITER == 0)
            {
                double elapsed = omp_get_wtime() - t_start;
                trace[tlen] = best->cost;
                timer[tlen] = elapsed;
                tlen++;
            }
        }

        // write per-thread log
        {
            char fname[64];
            sprintf(fname, "trace_thread_%d.csv", tid);
            FILE *fp = fopen(fname, "w");
            if (fp)
            {
                fprintf(fp, "time,best_cost\n");
                for (long long i = 0; i < tlen; i++)
                    fprintf(fp, "%.6f,%.10f\n", timer[i], trace[i]);
                fclose(fp);
            }
        }

        // update global best
        #pragma omp critical
        {
            if (best->cost < g_best->cost)
                qap_solution_copy(g_best, best);
        }

        free(trace);
        free(timer);
        qap_solution_free(cur);
        qap_solution_free(best);
        free_sprng(stream);
    } // end parallel

    // =============== PRINT GLOBAL BEST ===============

    printf("\nGLOBAL BEST COST = %.10f\n", g_best->cost);
    printf("Permutation:\n");
    for (size_t i = 0; i < n; i++)
        printf("%zu ", g_best->permutations[i]);
    printf("\n");

    qap_solution_free(g_best);
    qap_free(P);

    return 0;
}
