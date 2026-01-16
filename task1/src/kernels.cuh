#pragma once

#include <stddef.h>

// Each thread sums elems_per_thread values from input and writes one output value.
__global__ void kernel_contiguous(const float* input,
                                  float* output,
                                  size_t n,
                                  int elems_per_thread,
                                  int num_threads);

// Each thread sums elems_per_thread values with stride = num_threads.
__global__ void kernel_strided(const float* input,
                               float* output,
                               size_t n,
                               int elems_per_thread,
                               int num_threads);
