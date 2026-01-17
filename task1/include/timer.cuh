#pragma once

#include <cuda_runtime.h>

typedef struct {
    cudaEvent_t start;
    cudaEvent_t stop;
} CudaTimer;

// CUDA-Event basierter Timer fuer Kernel-Laufzeiten.
static inline void timerCreate(CudaTimer* t) {
    cudaEventCreate(&t->start);
    cudaEventCreate(&t->stop);
}

static inline void timerDestroy(CudaTimer* t) {
    cudaEventDestroy(t->start);
    cudaEventDestroy(t->stop);
}

static inline void timerStart(CudaTimer* t) {
    cudaEventRecord(t->start, 0);
}

static inline float timerStopMs(CudaTimer* t) {
    cudaEventRecord(t->stop, 0);
    cudaEventSynchronize(t->stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t->start, t->stop);
    return ms;
}
