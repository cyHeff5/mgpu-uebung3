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

#define CHECK_CUDA(call) checkCuda((call), #call)

typedef struct {
    cudaEvent_t start;
    cudaEvent_t stop;
} CudaTimer;

static inline void timerCreate(CudaTimer* t) {
    CHECK_CUDA(cudaEventCreate(&t->start));
    CHECK_CUDA(cudaEventCreate(&t->stop));
}

static inline void timerDestroy(CudaTimer* t) {
    CHECK_CUDA(cudaEventDestroy(t->start));
    CHECK_CUDA(cudaEventDestroy(t->stop));
}

static inline void timerStart(CudaTimer* t) {
    CHECK_CUDA(cudaEventRecord(t->start, 0));
}

static inline float timerStopMs(CudaTimer* t) {
    CHECK_CUDA(cudaEventRecord(t->stop, 0));
    CHECK_CUDA(cudaEventSynchronize(t->stop));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, t->start, t->stop));
    return ms;
}
