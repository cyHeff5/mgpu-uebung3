#include <stdio.h>
#include <stdlib.h>

#include "kernels.cuh"
#include "utils.cuh"

static void initInput(float* data, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        data[i] = 1.0f;
    }
}

static float runKernelContiguous(const float* d_in,
                                 float* d_out,
                                 size_t n,
                                 int elems_per_thread,
                                 int num_threads,
                                 int block_size) {
    int grid = (num_threads + block_size - 1) / block_size;

    CudaTimer t;
    timerCreate(&t);
    timerStart(&t);
    kernel_contiguous<<<grid, block_size>>>(d_in, d_out, n, elems_per_thread, num_threads);
    CHECK_CUDA(cudaGetLastError());
    float ms = timerStopMs(&t);
    timerDestroy(&t);

    return ms;
}

static float runKernelStrided(const float* d_in,
                              float* d_out,
                              size_t n,
                              int elems_per_thread,
                              int num_threads,
                              int block_size) {
    int grid = (num_threads + block_size - 1) / block_size;

    CudaTimer t;
    timerCreate(&t);
    timerStart(&t);
    kernel_strided<<<grid, block_size>>>(d_in, d_out, n, elems_per_thread, num_threads);
    CHECK_CUDA(cudaGetLastError());
    float ms = timerStopMs(&t);
    timerDestroy(&t);

    return ms;
}

int main(void) {
    const int num_threads = 20000;
    const int elems_per_thread = 5000;
    const size_t n = (size_t)num_threads * (size_t)elems_per_thread;

    float* h_in = (float*)malloc(n * sizeof(float));
    float* h_out = (float*)malloc((size_t)num_threads * sizeof(float));
    if (!h_in || !h_out) {
        fprintf(stderr, "Host allocation failed\n");
        return EXIT_FAILURE;
    }

    initInput(h_in, n);

    float* d_in = NULL;
    float* d_out = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_in, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_out, (size_t)num_threads * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice));

    for (int k = 0; ; ++k) {
        int block_size = 32;
        for (int i = 0; i < k; ++i) {
            block_size *= 5;
        }
        if (block_size > 1024) {
            break;
        }

        // Warmup run (not timed) to avoid JIT/cold-start effects.
        int grid = (num_threads + block_size - 1) / block_size;
        kernel_contiguous<<<grid, block_size>>>(d_in, d_out, n, elems_per_thread, num_threads);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        kernel_strided<<<grid, block_size>>>(d_in, d_out, n, elems_per_thread, num_threads);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        float ms_a = runKernelContiguous(d_in, d_out, n, elems_per_thread, num_threads, block_size);
        CHECK_CUDA(cudaMemcpy(h_out, d_out, (size_t)num_threads * sizeof(float), cudaMemcpyDeviceToHost));
        printf("k=%d block_size=%d Variant A (contiguous) ms: %.3f\n", k, block_size, ms_a);

        float ms_b = runKernelStrided(d_in, d_out, n, elems_per_thread, num_threads, block_size);
        CHECK_CUDA(cudaMemcpy(h_out, d_out, (size_t)num_threads * sizeof(float), cudaMemcpyDeviceToHost));
        printf("k=%d block_size=%d Variant B (strided)    ms: %.3f\n", k, block_size, ms_b);
    }

    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_in));
    free(h_out);
    free(h_in);

    return 0;
}
