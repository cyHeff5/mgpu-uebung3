#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime.h>

#include "cuda_utils.cuh"
#include "kernels.cuh"
#include "matrix.h"
#include "timer.cuh"

static void usage(const char* prog) {
    printf("Usage: %s N [block_size]\n", prog);
}

static int is_pow2(int v) {
    return v > 0 && (v & (v - 1)) == 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    if (n <= 0) {
        fprintf(stderr, "N must be > 0\n");
        return 1;
    }

    int block_size = 256;
    // Jede Messung mehrfach ausfuehren, damit der Mittelwert stabiler ist.
    const int repeats = 5;
    if (argc >= 3) {
        block_size = atoi(argv[2]);
        if (block_size <= 0) {
            block_size = 256;
        }
    }
    if (!is_pow2(block_size)) {
        printf("Block size %d ist keine Zweierpotenz, setze 256.\n", block_size);
        block_size = 256;
    }

    // Host-Speicher fuer Matrix und Vektoren.
    float* h_a = matrix_alloc((size_t)n);
    float* h_x = vector_alloc((size_t)n);
    float* h_y = vector_alloc((size_t)n);
    if (!h_a || !h_x || !h_y) {
        fprintf(stderr, "Host allocation failed\n");
        matrix_free(h_a);
        vector_free(h_x);
        vector_free(h_y);
        return 1;
    }

    matrix_init_upper(h_a, (size_t)n);
    vector_init(h_x, (size_t)n);

    // Device-Speicher anlegen und Daten kopieren.
    float* d_a = NULL;
    float* d_x = NULL;
    float* d_y = NULL;
    size_t bytes_a = (size_t)n * (size_t)n * sizeof(float);
    size_t bytes_v = (size_t)n * sizeof(float);
    CHECK_CUDA(cudaMalloc((void**)&d_a, bytes_a));
    CHECK_CUDA(cudaMalloc((void**)&d_x, bytes_v));
    CHECK_CUDA(cudaMalloc((void**)&d_y, bytes_v));
    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, bytes_v, cudaMemcpyHostToDevice));

    // Ein Block pro Zeile.
    int grid = n;

    CudaTimer t;
    if (!timer_create(&t)) {
        fprintf(stderr, "Timer create failed\n");
        return 1;
    }

    double mean = 0.0;
    double m2 = 0.0;
    for (int r = 0; r < repeats; ++r) {
        if (!timer_start(&t)) {
            fprintf(stderr, "Timer start failed\n");
            return 1;
        }
        // Shared-Memory-Groesse: eine Float-Summe pro Thread.
        matvec_upper<<<grid, block_size, block_size * sizeof(float)>>>(d_a, d_x, d_y, n);
        CHECK_CUDA(cudaGetLastError());
        float ms = timer_stop_ms(&t);
        if (ms < 0.0f) {
            fprintf(stderr, "Timer stop failed\n");
            return 1;
        }
        double delta = (double)ms - mean;
        mean += delta / (double)(r + 1);
        double delta2 = (double)ms - mean;
        m2 += delta * delta2;
    }
    double variance = (repeats > 1) ? (m2 / (double)(repeats - 1)) : 0.0;
    float ms = (float)mean;
    float std_ms = (float)sqrt(variance);

    CHECK_CUDA(cudaMemcpy(h_y, d_y, bytes_v, cudaMemcpyDeviceToHost));

    printf("N=%d block_size=%d\n", n, block_size);
    printf("CUDA kernel time: %.3f ms +/- %.3f\n", ms, std_ms);

    timer_destroy(&t);
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_a));
    vector_free(h_y);
    vector_free(h_x);
    matrix_free(h_a);

    return 0;
}
