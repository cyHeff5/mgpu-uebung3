#ifndef TASK3_TIMER_H
#define TASK3_TIMER_H

#include <cuda_runtime.h>

typedef struct CudaTimer {
    cudaEvent_t start;
    cudaEvent_t stop;
} CudaTimer;

int timer_create(CudaTimer* t);
int timer_destroy(CudaTimer* t);
int timer_start(CudaTimer* t);
float timer_stop_ms(CudaTimer* t);

#endif  // TASK3_TIMER_H
