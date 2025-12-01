#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

#include "sprng.h"   // SPRNG 2.0

#define IDX(n,i,j) ((i) * (n) + (j))
#define SEED 12345

#define LOG_EVERY_ITER 100

typedef struct
{
    size_t n;
    const double *flow;
    const double *distance;
} QAPProblem;

typedef struct
{
    size_t n;
    size_t *permutations;
    double cost;
} QAPSolution;

// ==================== LOAD ====================

QAPProblem *qap_load_from_file(const char *path)
{
    FILE *fp = fopen(path, "r");
    if (!fp) return NULL;

    size_t n;
    if (fscanf(fp, "%zu", &n) != 1)
    {
        fclose(fp);
        return NULL;
    }

    double *flow = malloc(n * n * sizeof(double));
    double *dist = malloc(n * n * sizeof(double));

    if (!flow || !dist)
    {
        free(flow); free(dist); fclose(fp); 
        return NULL;
    }

    for (size_t i = 0; i < n*n; i++)
        if (fscanf(fp, "%lf", &flow[i]) != 1)
        {
            free(flow); free(dist); fclose(fp);
            return NULL;
        }

    for (size_t i = 0; i < n*n; i++)
        if (fscanf(fp, "%lf", &dist[i]) != 1)
        {
            free(flow); free(dist); fclose(fp);
            return NULL;
        }

    fclose(fp);

    QAPProblem *p = malloc(sizeof(QAPProblem));
    p->n = n;
    p->flow = flow;
    p->distance = dist;
    return p;
}

void qap_free(QAPProblem *p)
{
    if (!p) return;
    free((void*)p->flow);
    free((void*)p->distance);
    free(p);
}

// ==================== EVAL ====================

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

// ==================== DELTA ====================

double qap_delta_swap(const QAPProblem *p, const QAPSolution *s, size_t i, size_t j)
{
    if (i == j) return 0.0;

    if (i > j) { size_t t=i; i=j; j=t; }

    size_t n = p->n;

    size_t pi = s->permutations[i];
    size_t pj = s->permutations[j];

    const double *F = p->flow;
    const double *D = p->distance;

    double delta = 0.0;

    for (size_t k = 0; k < n; k++)
    {
        if (k == i || k == j) continue;

        size_t pk = s->permutations[k];

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

// ==================== SOLUTION ====================

QAPSolution *qap_solution_alloc(size_t n)
{
    QAPSolution *s = malloc(sizeof(QAPSolution));
    s->n = n;
    s->permutations = malloc(n * sizeof(size_t));
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
    for (size_t i = 0; i < src->n; i++)
        dst->permutations[i] = src->permutations[i];
}

void qap_solution_random_init(const QAPProblem *p, QAPSolution *s, int *stream)
{
    size_t n = p->n;

    for (size_t i = 0; i < n; i++)
        s->permutations[i] = i;

    for (size_t i = n - 1; i > 0; i--)
    {
        size_t j = (size_t)(sprng(stream) * (i+1));
        size_t tmp = s->permutations[i];
        s->permutations[i] = s->permutations[j];
        s->permutations[j] = tmp;
    }

    s->cost = qap_eval(p, s);
}

// ==================== GLOBAL BEST ====================

static QAPSolution *g_best = NULL;
static pthread_mutex_t g_best_mutex = PTHREAD_MUTEX_INITIALIZER;

void update_global_best(const QAPSolution *cand)
{
    pthread_mutex_lock(&g_best_mutex);

    if (!g_best)
    {
        g_best = qap_solution_alloc(cand->n);
        qap_solution_copy(g_best, cand);
        fprintf(stderr, "[GLOBAL] init best = %.10f\n", g_best->cost);
    }
    else if (cand->cost < g_best->cost)
    {
        qap_solution_copy(g_best, cand);
        fprintf(stderr, "[GLOBAL] improved best = %.10f\n", g_best->cost);
    }

    pthread_mutex_unlock(&g_best_mutex);
}

// ==================== THREAD ARGS ====================

typedef struct
{
    const QAPProblem *problem;
    long long n_iters;
    int thread_id;
    int n_threads;
    double T0;
    double alpha;
    long long restart_period;
    int shake_moves;

    // === LOGGING ===
    double *trace;
    double *timer;
    long long trace_len;

} SAThreadArgs;

// ==================== SA THREAD ====================

void *sa_thread_func(void *arg)
{
    SAThreadArgs *a = (SAThreadArgs*)arg;
    const QAPProblem *p = a->problem;
    size_t n = p->n;

    int *stream = init_sprng(DEFAULT_RNG_TYPE,
                             a->thread_id,
                             a->n_threads,
                             SEED,
                             SPRNG_DEFAULT);

    QAPSolution *cur  = qap_solution_alloc(n);
    QAPSolution *best = qap_solution_alloc(n);

    qap_solution_random_init(p, cur, stream);
    qap_solution_copy(best, cur);
    update_global_best(best);

    double T = a->T0;

    // ===== TIMER START =====
    struct timespec ts_start;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    for (long long iter = 0; iter < a->n_iters; iter++)
    {
        size_t i = (size_t)(sprng(stream) * n);
        size_t j = (size_t)(sprng(stream) * n);

        if (i != j)
        {
            double d = qap_delta_swap(p, cur, i, j);

            int accept = 0;
            if (d < 0) accept = 1;
            else if (sprng(stream) < exp(-d/T)) accept = 1;

            if (accept)
            {
                qap_apply_swap(cur, i, j, d);

                if (cur->cost < best->cost)
                {
                    qap_solution_copy(best, cur);
                    update_global_best(best);
                }
            }
        }

        if ((iter % LOG_EVERY_ITER) == 0)
        {
            struct timespec ts_now;
            clock_gettime(CLOCK_MONOTONIC, &ts_now);

            double elapsed =
                (ts_now.tv_sec  - ts_start.tv_sec) +
                (ts_now.tv_nsec - ts_start.tv_nsec) / 1e9;

            a->trace[a->trace_len] = cur->cost;
            a->timer[a->trace_len] = elapsed;
            a->trace_len++;
        }

        if (((iter % n)==0) && T>1e-8) 
            T *= a->alpha;

        if (a->restart_period>0 && iter>0 && (iter % a->restart_period)==0)
        {
            pthread_mutex_lock(&g_best_mutex);

            if (g_best)
            {
                qap_solution_copy(cur, g_best);
                qap_solution_copy(best, g_best);
            }

            pthread_mutex_unlock(&g_best_mutex);

            for (int s = 0; s < a->shake_moves; s++)
            {
                size_t si = (size_t)(sprng(stream)*n);
                size_t sj = (size_t)(sprng(stream)*n);
                if (si == sj) continue;

                double dd = qap_delta_swap(p, cur, si, sj);
                qap_apply_swap(cur, si, sj, dd);
            }

            T = a->T0;
        }
    }

    update_global_best(best);

    qap_solution_free(cur);
    qap_solution_free(best);
    free_sprng(stream);
    return NULL;
}

// ==================== MAIN ====================

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr,
                "Usage: %s instance.dat [nthreads] [iters] [restart] [shake] [repeats]\n",
                argv[0]);
        return 1;
    }

    const char *path         = (argc > 1 ? argv[1] : NULL);
    int nthreads             = (argc > 2 ? atoi(argv[2]) : 4);
    long long iters          = (argc > 3 ? atoll(argv[3]) : 100000);
    long long restart_period = (argc > 4 ? atoll(argv[4]) : 20000);
    int shake_moves          = (argc > 5 ? atoi(argv[5]) : 5);
    int repeats              = (argc > 6 ? atoi(argv[6]) : 1);

    QAPProblem *p = qap_load_from_file(path);
    if (!p)
    {
        fprintf(stderr, "Failed to load QAP instance\n");
        return 1;
    }

    long long approx_logs = iters / LOG_EVERY_ITER + 10;

    // === массивы для усреднения ===
    double *best_cost = malloc(approx_logs * sizeof(double));
    double *sum_time  = calloc(approx_logs, sizeof(double));

    for (long long i = 0; i < approx_logs; i++)
        best_cost[i] = 1e300;  // очень большое число

    pthread_t *threads;
    SAThreadArgs *args;

    // === ПОВТОРЫ R раз ===
    for (int r = 0; r < repeats; r++)
    {
        fprintf(stderr, "\n=== REPEAT %d / %d ===\n", r + 1, repeats);

        threads = malloc(nthreads * sizeof(pthread_t));
        args    = malloc(nthreads * sizeof(SAThreadArgs));

        // обнулить глобальный best между повторами
        pthread_mutex_lock(&g_best_mutex);
        if (g_best)
        {
            qap_solution_free(g_best);
            g_best = NULL;
        }
        pthread_mutex_unlock(&g_best_mutex);

        // запуск потоков
        for (int t = 0; t < nthreads; t++)
        {
            args[t].problem        = p;
            args[t].n_iters        = iters;
            args[t].thread_id      = t;
            args[t].n_threads      = nthreads;
            args[t].T0             = 1000.0;
            args[t].alpha          = 0.995;
            args[t].restart_period = restart_period;
            args[t].shake_moves    = shake_moves;

            args[t].trace = malloc(approx_logs * sizeof(double));
            args[t].timer = malloc(approx_logs * sizeof(double));
            args[t].trace_len = 0;

            pthread_create(&threads[t], NULL, sa_thread_func, &args[t]);
        }

        // ждём окончания
        for (int t = 0; t < nthreads; t++)
            pthread_join(threads[t], NULL);

        // аккумуляция итогов текущего повтора
        for (int t = 0; t < nthreads; t++)
        {
            for (long long i = 0; i < args[t].trace_len; i++)
            {
                // берём МИНИМУМ стоимости (best-so-far)
                if (args[t].trace[i] < best_cost[i])
                    best_cost[i] = args[t].trace[i];

                // время усредняем
                sum_time[i] += args[t].timer[i];
            }

            free(args[t].trace);
            free(args[t].timer);
        }

        free(threads);
        free(args);
    }

    // === вычисление окончательного усреднённого времени ===
    double total = (double)(repeats * nthreads);

    FILE *avg = fopen("average_trace.csv", "w");
    fprintf(avg, "time,best_cost\n");

    for (long long i = 0; i < approx_logs; i++)
    {
        double avg_time = sum_time[i] / total;
        fprintf(avg, "%.6f,%.10f\n", avg_time, best_cost[i]);
    }

    fclose(avg);

    printf("Saved averaged trace to average_trace.csv\n");

    // вывести лучший последний результат
    if (g_best)
    {
        printf("Best cost = %.10f\n", g_best->cost);
        printf("Permutation:\n");
        for (size_t i = 0; i < g_best->n; i++)
            printf("%zu ", g_best->permutations[i]);
        printf("\n");
    }

    free(best_cost);
    free(sum_time);

    qap_free(p);
    if (g_best) qap_solution_free(g_best);

    return 0;
}
