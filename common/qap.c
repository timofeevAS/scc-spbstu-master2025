#include "qap.h"

#include <stdio.h>
#include <stdlib.h>

#define IDX(n,i,j) ((i) * (n) + (j))

QAPProblem *qap_load_from_file(const char *path)
{
    FILE *fp = fopen(path, "r");
    if (!fp) 
    {
        return NULL;
    }

    size_t n;
    if (fscanf(fp, "%zu", &n) != 1) 
    { 
        fclose(fp); 
        return NULL; 
    }

    double *flow = (double*)malloc(n * n * sizeof(double));
    double *dist = (double*)malloc(n * n * sizeof(double));
    if (!flow || !dist) 
    { 
        free(flow); 
        free(dist); 
        fclose(fp); 
        return NULL; 
    }

    // read (n x n) values: flow
    for (size_t i = 0; i < n*n; i++) 
    {
        if (fscanf(fp, "%lf", &flow[i]) != 1) 
        {
             free(flow); free(dist); fclose(fp); 
             return NULL; 
        }
    }
    // read (n x n) values: distance
    for (size_t i = 0; i < n*n; i++) 
    {
        if (fscanf(fp, "%lf", &dist[i]) != 1) 
        { 
            free(flow); 
            free(dist); 
            fclose(fp); 
            return NULL; 
        }
    }

    fclose(fp);

    QAPProblem *p = (QAPProblem*)malloc(sizeof(QAPProblem));
    if (!p) 
    { 
        free(flow); 
        free(dist); 
        return NULL; 
    }

    p->n = n;
    // NOTE: Remind to free this memory in qap_free().
    p->flow = flow;
    p->distance = dist;
    return p;
}

void qap_free(QAPProblem *p) 
{
    if (!p)
    {
        return;
    }

    free((void*)p->flow);
    free((void*)p->distance);
    free(p);
}

double qap_eval(const QAPProblem *p, const QAPSolution *s)
{
    double cost = 0.0;
    for (size_t i = 0; i < p->n; i++) 
    {
        size_t li = s->permutations[i];
        for (size_t j = 0; j < p->n; j++) 
        {
            size_t lj = s->permutations[j];
            cost += p->flow[IDX(p->n,i,j)] * p->distance[IDX(p->n,li,lj)];
        }
    }
    
    return cost;
}

double qap_delta_swap(const QAPProblem *p, const QAPSolution *s, size_t i, size_t j)
{
    if (i == j) 
    {
        return 0.0;
    }

    if (i > j)
     { 
        
        size_t t = i; 
        i = j; 
        j = t; 
    }

    const size_t n  = p->n;
    const size_t pi = s->permutations[i];
    const size_t pj = s->permutations[j];

    const double *F = p->flow;
    const double *D = p->distance;

    double delta = 0.0;

    // вклад всех k != i, j
    for (size_t k = 0; k < n; k++) {
        if (k == i || k == j)
        {
            continue;
        }
        const size_t pk = s->permutations[k];

        delta += (F[IDX(n,i,k)] - F[IDX(n,j,k)]) * (D[IDX(n,pj,pk)] - D[IDX(n,pi,pk)])
               + (F[IDX(n,k,i)] - F[IDX(n,k,j)]) * (D[IDX(n,pk,pj)] - D[IDX(n,pk,pi)]);
    }

    delta += (F[IDX(n,i,i)] - F[IDX(n,j,j)]) * (D[IDX(n,pj,pj)] - D[IDX(n,pi,pi)])
           + (F[IDX(n,i,j)] - F[IDX(n,j,i)]) * (D[IDX(n,pj,pi)] - D[IDX(n,pi,pj)]);

    return delta;
}

void qap_apply_swap(QAPSolution *s, size_t i, size_t j, double delta)
{
    if (i == j)
    {
        return;
    }
    size_t tmp = s->permutations[i];
    s->permutations[i] = s->permutations[j];
    s->permutations[j] = tmp;
    s->cost += delta;
}
