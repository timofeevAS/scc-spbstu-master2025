#ifndef QAP_H
#define QAP_H

#include <stddef.h>
#include <ctype.h>

typedef struct
{
    size_t n;
    const double *flow;      // matrix of flows (n x n)
    const double *distance;  // matrix of distance (n x n)
} QAPProblem;

typedef struct
{
    size_t n;
    size_t *permutations;
    double cost;
} QAPSolution;

// Load problem instance from a simple text file:
// n
// f00 f01 ... f0(n-1)
// ...
// f(n-1)0 ... f(n-1)(n-1)
// d00 d01 ... d0(n-1)
// ...
// d(n-1)0 ... d(n-1)(n-1)
QAPProblem *qap_load_from_file(const char *path);

// Free memory allocated for flow/distance matrices and struct.
void qap_free(QAPProblem *p);

// Evaluate full cost for current permutation.
double qap_eval(const QAPProblem *p, const QAPSolution *s);

// Compute delta for swapping two indices i, j (O(n)).
double qap_delta_swap(const QAPProblem *p, const QAPSolution *s, size_t i, size_t j);

// Apply swap and update cost. Synchornize cost field.
void qap_apply_swap(QAPSolution *s, size_t i, size_t j, double delta);

#endif // QAP_H