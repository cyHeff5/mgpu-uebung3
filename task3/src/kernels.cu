#include "kernels.cuh"

__global__ void matvec_upper(const float* a, const float* x, float* y, int n) {
    int row = blockIdx.x;
    if (row >= n) {
        return;
    }

    float sum = 0.0f;
    int base = row * n;
    for (int j = row + threadIdx.x; j < n; j += blockDim.x) {
        sum += a[base + j] * x[j];
    }

    extern __shared__ float sdata[];
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        y[row] = sdata[0];
    }
}
