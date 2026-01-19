#ifndef TASK3_TIMER_CUH
#define TASK3_TIMER_CUH

#include <cuda_runtime.h>

typedef struct CudaTimer {
    cudaEvent_t start;
    cudaEvent_t stop;
} CudaTimer;

// CUDA-Event-basierter Timer fuer Kernelzeiten.
static inline void timer_create(CudaTimer* t) {
    cudaEventCreate(&t->start);
    cudaEventCreate(&t->stop);
}

static inline void timer_destroy(CudaTimer* t) {
    cudaEventDestroy(t->start);
    cudaEventDestroy(t->stop);
}

static inline void timer_start(CudaTimer* t) {
    cudaEventRecord(t->start, 0);
}

static inline float timer_stop_ms(CudaTimer* t) {
    cudaEventRecord(t->stop, 0);
    cudaEventSynchronize(t->stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t->start, t->stop);
    return ms;
}

#endif  // TASK3_TIMER_CUH
