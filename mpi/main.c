#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "sprng.h"

#include "../common/qap.h"

#define BOTTOM_TEMPERATURE_BOUND 1e-6
#define LOG_EVERY_ITER 1000


// ---------------- SPRNG wrappers ----------------

static inline double rng_uniform(int *stream)
{
    return sprng(stream); // [0,1)
}

static inline size_t rng_randint(int *stream, size_t n)
{
    return (size_t)(sprng(stream) * n);
}


// ---------------- Solution helpers ----------------

static void sol_alloc(QAPSolution *s, size_t n)
{
    s->n = n;
    s->permutations = malloc(n * sizeof(size_t));
    s->cost = 0.0;
}

static void sol_free(QAPSolution *s)
{
    free(s->permutations);
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
        size_t j = rng_randint(stream, i + 1);
        size_t t = s->permutations[i];
        s->permutations[i] = s->permutations[j];
        s->permutations[j] = t;
    }
}

static void sol_copy(const QAPSolution *src, QAPSolution *dst)
{
    memcpy(dst->permutations, src->permutations, src->n * sizeof(size_t));
    dst->cost = src->cost;
}



// ----------------------- MAIN -----------------------

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc < 2)
    {
        if (rank == 0)
            fprintf(stderr, "Usage: mpirun -np N ./sa_mpi instance.dat [iters] [random_start]\n");
        MPI_Finalize();
        return 1;
    }

    const char *path = argv[1];
    long long iters = (argc > 2 ? atoll(argv[2]) : 100000);
    int random_start = (argc > 3 ? atoi(argv[3]) : 0);   // <=== CONTROL RANDOM START

    if (rank == 0)
    {
        printf("Running MPI SA with %d processes\n", nprocs);
        printf("random_start = %d\n", random_start);
    }

    // Load QAP
    QAPProblem *P = qap_load_from_file(path);
    if (!P)
    {
        if (rank == 0)
            fprintf(stderr, "Failed to load QAP\n");
        MPI_Finalize();
        return 1;
    }

    size_t n = P->n;

    // SA params EXACTLY like single-thread version
    double T0    = 300000.0;
    double alpha = 0.9992;


    // ------------ SPRNG init with correct independent streams ------------
    int *stream = init_sprng(
        DEFAULT_RNG_TYPE,
        rank,     // stream ID
        nprocs,   // total streams
        12345,    // seed
        SPRNG_DEFAULT
    );

    // allocate solutions
    QAPSolution current, best;
    sol_alloc(&current, n);
    sol_alloc(&best, n);

    // initialize solution
    sol_identity(&current);

    if (random_start)
        sol_shuffle(&current, stream);

    current.cost = qap_eval(P, &current);
    sol_copy(&current, &best);


    // logging arrays
    long long max_logs = iters / LOG_EVERY_ITER + 5;
    double *trace = malloc(max_logs * sizeof(double));
    double *timelog = malloc(max_logs * sizeof(double));
    long long tlen = 0;

    double T = T0;

    double t_start = MPI_Wtime();


    // ---------------------- SA loop ----------------------

    for (long long it = 0; it < iters; it++)
    {
        size_t i = rng_randint(stream, n);
        size_t j = rng_randint(stream, n);
        if (i == j) continue;

        double delta = qap_delta_swap(P, &current, i, j);

        int accept = 0;

        if (delta < 0.0)
            accept = 1;
        else {
            double u = rng_uniform(stream);
            double prob = (T > 0.0 ? exp(-delta / T) : 0.0);
            if (u < prob)
                accept = 1;
        }

        if (accept)
        {
            qap_apply_swap(&current, i, j, delta);
            if (current.cost < best.cost)
                sol_copy(&current, &best);
        }

        T *= alpha;
        if (T < BOTTOM_TEMPERATURE_BOUND)
            T = BOTTOM_TEMPERATURE_BOUND;

        if ((it % LOG_EVERY_ITER) == 0)
        {
            double elapsed = MPI_Wtime() - t_start;
            if (tlen < max_logs)
            {
                trace[tlen] = best.cost;
                timelog[tlen] = elapsed;
                tlen++;
            }
        }
    }

    double my_best_cost = best.cost;


    // ---------------------- Write trace log ----------------------

    {
        char fname[256];
        sprintf(fname, "trace_rank_%d.csv", rank);
        FILE *fp = fopen(fname, "w");
        fprintf(fp, "time,best_cost\n");
        for (long long i = 0; i < tlen; i++)
            fprintf(fp, "%.6f,%.10f\n", timelog[i], trace[i]);
        fclose(fp);
    }


    // ---------------------- Gather global best ----------------------

    double *all_costs = NULL;

    if (rank == 0)
        all_costs = malloc(nprocs * sizeof(double));

    MPI_Gather(&my_best_cost, 1, MPI_DOUBLE,
               all_costs,     1, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    int best_rank = 0;

    if (rank == 0)
    {
        double global_best = all_costs[0];
        best_rank = 0;

        for (int r = 1; r < nprocs; r++)
        {
            if (all_costs[r] < global_best)
            {
                global_best = all_costs[r];
                best_rank = r;
            }
        }

        printf("\nGLOBAL BEST COST = %.10f (from rank %d)\n",
               global_best, best_rank);

        free(all_costs);
    }

    MPI_Bcast(&best_rank, 1, MPI_INT, 0, MPI_COMM_WORLD);


    // ---------------------- Send best permutation to rank 0 ----------------------

    int *perm_buf = NULL;
    if (rank == 0)
        perm_buf = malloc(n * sizeof(int));

    if (rank == best_rank)
    {
        int *tmp = malloc(n * sizeof(int));
        for (size_t i = 0; i < n; i++)
            tmp[i] = (int)best.permutations[i];

        MPI_Send(tmp, n, MPI_INT, 0, 0, MPI_COMM_WORLD);
        free(tmp);
    }

    if (rank == 0)
    {
        MPI_Recv(perm_buf, n, MPI_INT, best_rank, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);

        printf("Permutation:\n");
        for (size_t i = 0; i < n; i++)
            printf("%d ", perm_buf[i]);
        printf("\n");

        free(perm_buf);
    }


    // cleanup
    sol_free(&current);
    sol_free(&best);
    free(trace);
    free(timelog);
    free_sprng(stream);
    qap_free(P);

    MPI_Finalize();
    return 0;
}
