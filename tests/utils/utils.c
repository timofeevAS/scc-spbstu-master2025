#include "utils.h"

#include <math.h>

int double_eq(double a, double b, double eps)
{
    return fabs(a - b) < eps;
}

int double_array_eq(const double *a, const double *b, size_t size, double eps)
{
    if (a == NULL || b == NULL)
    {
        return 0;
    }

    for (size_t i = 0; i < size; i++) 
    {
        if (!double_eq(a[i], b[i], eps))
        {
            return 0;
        }
    }

    return 1;
}
