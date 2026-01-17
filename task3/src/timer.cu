#include "timer.h"

int timer_create(CudaTimer* t) {
    if (cudaEventCreate(&t->start) != cudaSuccess) {
        return 0;
    }
    if (cudaEventCreate(&t->stop) != cudaSuccess) {
        cudaEventDestroy(t->start);
        return 0;
    }
    return 1;
}

int timer_destroy(CudaTimer* t) {
    if (cudaEventDestroy(t->start) != cudaSuccess) {
        return 0;
    }
    if (cudaEventDestroy(t->stop) != cudaSuccess) {
        return 0;
    }
    return 1;
}

int timer_start(CudaTimer* t) {
    if (cudaEventRecord(t->start, 0) != cudaSuccess) {
        return 0;
    }
    return 1;
}

float timer_stop_ms(CudaTimer* t) {
    if (cudaEventRecord(t->stop, 0) != cudaSuccess) {
        return -1.0f;
    }
    if (cudaEventSynchronize(t->stop) != cudaSuccess) {
        return -1.0f;
    }
    float ms = 0.0f;
    if (cudaEventElapsedTime(&ms, t->start, t->stop) != cudaSuccess) {
        return -1.0f;
    }
    return ms;
}
