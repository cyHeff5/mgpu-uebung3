#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

static inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Makro fuer kurze CUDA-Fehlerpruefung.
#define CHECK_CUDA(call) checkCuda((call), #call)
