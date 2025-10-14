#ifndef TESTS_UTILS_H
#define TESTS_UTILS_H

#include <stddef.h>

// Compare two double digits with accuracy eps.
int double_eq(double a, double b, double eps);
int double_array_eq(const double *a, const double *b, size_t size, double eps);

#endif // TESTS_UTILS_Hs