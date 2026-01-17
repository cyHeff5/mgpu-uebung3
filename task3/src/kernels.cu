#include "kernels.cuh"

__global__ void matvec_upper(const float* a, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    float sum = 0.0f;
    int row = i * n;
    for (int j = i; j < n; ++j) {
        sum += a[row + j] * x[j];
    }
    y[i] = sum;
}
