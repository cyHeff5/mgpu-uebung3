#ifndef TASK2_MATVEC_H
#define TASK2_MATVEC_H

#include <stddef.h>

void matvec_seq(const float* a, const float* x, float* y, size_t n);
void matvec_omp(const float* a, const float* x, float* y, size_t n, int num_threads);

#endif  // TASK2_MATVEC_H
