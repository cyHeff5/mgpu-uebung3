#pragma once

#include <stddef.h>

// Variante A: zusammenhaengender Block pro Thread.
__global__ void kernel_contiguous(const float* input,
                                  float* output,
                                  size_t n,
                                  int elems_per_thread,
                                  int num_threads);

// Variante B: strided Zugriff mit stride = num_threads.
__global__ void kernel_strided(const float* input,
                               float* output,
                               size_t n,
                               int elems_per_thread,
                               int num_threads);
