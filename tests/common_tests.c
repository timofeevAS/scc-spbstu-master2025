#include <stdio.h>
#include <assert.h>

#include "utils/utils.h"

#include "../common/qap.h"


int main(void) {
    // Create temporary file with data.
    FILE *f = fopen("tmp_qap.txt", "w");
    assert(f && "cannot create temp file");
    fprintf(f,
        "3\n"
        "0 1 2\n"
        "1 0 3\n"
        "2 3 0\n"
        "0 5 2\n"
        "5 0 4\n"
        "2 4 0\n"
    );
    fclose(f);
    
    // Check qap_load_from_file().
    QAPProblem *problem = qap_load_from_file("tmp_qap.txt");
    assert(problem && "Failed to load QAP file");

    double expected_flow[9] = 
    {
        0, 1, 2,
        1, 0, 3,
        2, 3, 0
    };
    
    double expected_dist[9] = 
    {
        0, 5, 2,
        5, 0, 4,
        2, 4, 0
    };

    assert(double_array_eq(expected_flow, problem->flow, problem->n, 1e-9));
    assert(double_array_eq(expected_dist, problem->distance, problem->n, 1e-9));

    // Check qap_cost().
    QAPSolution test_solution;
    test_solution.n = 3;
    test_solution.permutations = (size_t[]){0, 1, 2};
    test_solution.cost = qap_eval(problem, &test_solution);
    assert(double_eq(test_solution.cost, 42, 1e-9));

    // Check qap_swap_cost().
    assert(double_eq(qap_delta_swap(problem, &test_solution, 0, 2), 4, 1e-9));

    // Check qap_apply_swap().
    qap_apply_swap(&test_solution, 0, 2, qap_delta_swap(problem, &test_solution, 0, 2));
    assert(double_eq(test_solution.cost, 46, 1e-9));

    qap_free(problem);
    remove("tmp_qap.txt");

    return 0;
}
