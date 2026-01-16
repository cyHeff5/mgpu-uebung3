#include "kernels.cuh"

__global__ void kernel_contiguous(const float* input,
                                  float* output,
                                  size_t n,
                                  int elems_per_thread,
                                  int num_threads) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads) {
        return;
    }

    size_t start = (size_t)tid * (size_t)elems_per_thread;
    size_t end = start + (size_t)elems_per_thread;
    if (end > n) {
        end = n;
    }

    float sum = 0.0f;
    for (size_t i = start; i < end; ++i) {
        sum += input[i];
    }
    output[tid] = sum;
}

__global__ void kernel_strided(const float* input,
                               float* output,
                               size_t n,
                               int elems_per_thread,
                               int num_threads) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_threads) {
        return;
    }

    float sum = 0.0f;
    size_t stride = (size_t)num_threads;
    size_t idx = (size_t)tid;
    for (int j = 0; j < elems_per_thread; ++j) {
        size_t i = idx + (size_t)j * stride;
        if (i >= n) {
            break;
        }
        sum += input[i];
    }
    output[tid] = sum;
}
