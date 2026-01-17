#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../include/kernels.cuh"
#include "../include/cuda_utils.cuh"
#include "../include/timer.cuh"

static void initInput(float* data, size_t n) {
    // Einfacher Testinput: alle Werte sind 1.0f.
    for (size_t i = 0; i < n; ++i) {
        data[i] = 1.0f;
    }
}

static void runKernelContiguous(const float* d_in,
                                float* d_out,
                                size_t n,
                                int elems_per_thread,
                                int num_threads,
                                int block_size,
                                int repeats,
                                float* mean_ms,
                                float* stddev_ms) {
    // Grid-Groesse passend zur gewaehlten Blockgroesse berechnen.
    int grid = (num_threads + block_size - 1) / block_size;

    double mean = 0.0;
    double m2 = 0.0;
    for (int r = 0; r < repeats; ++r) {
        CudaTimer t;
        timerCreate(&t);
        timerStart(&t);
        // Kernel starten und die Laufzeit messen.
        kernel_contiguous<<<grid, block_size>>>(d_in, d_out, n, elems_per_thread, num_threads);
        CHECK_CUDA(cudaGetLastError());
        float ms = timerStopMs(&t);
        timerDestroy(&t);
        double delta = (double)ms - mean;
        mean += delta / (double)(r + 1);
        double delta2 = (double)ms - mean;
        m2 += delta * delta2;
    }

    double variance = (repeats > 1) ? (m2 / (double)(repeats - 1)) : 0.0;
    *mean_ms = (float)mean;
    *stddev_ms = (float)sqrt(variance);
}

static void runKernelStrided(const float* d_in,
                             float* d_out,
                             size_t n,
                             int elems_per_thread,
                             int num_threads,
                             int block_size,
                             int repeats,
                             float* mean_ms,
                             float* stddev_ms) {
    // Grid-Groesse passend zur gewaehlten Blockgroesse berechnen.
    int grid = (num_threads + block_size - 1) / block_size;

    double mean = 0.0;
    double m2 = 0.0;
    for (int r = 0; r < repeats; ++r) {
        CudaTimer t;
        timerCreate(&t);
        timerStart(&t);
        // Kernel starten und die Laufzeit messen.
        kernel_strided<<<grid, block_size>>>(d_in, d_out, n, elems_per_thread, num_threads);
        CHECK_CUDA(cudaGetLastError());
        float ms = timerStopMs(&t);
        timerDestroy(&t);
        double delta = (double)ms - mean;
        mean += delta / (double)(r + 1);
        double delta2 = (double)ms - mean;
        m2 += delta * delta2;
    }

    double variance = (repeats > 1) ? (m2 / (double)(repeats - 1)) : 0.0;
    *mean_ms = (float)mean;
    *stddev_ms = (float)sqrt(variance);
}

int main(void) {
    // Parameter aus der Aufgabenstellung.
    const int num_threads = 20000;
    const int elems_per_thread = 5000;
    const size_t n = (size_t)num_threads * (size_t)elems_per_thread;
    // Jede Messung mehrfach ausfuehren, damit der Mittelwert stabiler ist.
    const int repeats = 5;

    // Host-Speicher fuer Eingabe und Ergebnis.
    float* h_in = (float*)malloc(n * sizeof(float));
    float* h_out = (float*)malloc((size_t)num_threads * sizeof(float));
    if (!h_in || !h_out) {
        fprintf(stderr, "Host allocation failed\n");
        return EXIT_FAILURE;
    }

    initInput(h_in, n);

    // Device-Speicher anlegen und Input kopieren.
    float* d_in = NULL;
    float* d_out = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_in, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_out, (size_t)num_threads * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice));

    // Schleife ueber k: block_size = 32 * 5^k, bis zur CUDA-Grenze.
    for (int k = 0; ; ++k) {
        int block_size = 32;
        for (int i = 0; i < k; ++i) {
            block_size *= 5;
        }
        if (block_size > 1024) {
            break;
        }

        // Warmup, damit JIT und erste Speicherzugriffe die Messung nicht verfalschen.
        int grid = (num_threads + block_size - 1) / block_size;
        kernel_contiguous<<<grid, block_size>>>(d_in, d_out, n, elems_per_thread, num_threads);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        kernel_strided<<<grid, block_size>>>(d_in, d_out, n, elems_per_thread, num_threads);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        float mean_a = 0.0f;
        float std_a = 0.0f;
        runKernelContiguous(d_in, d_out, n, elems_per_thread, num_threads, block_size, repeats,
                            &mean_a, &std_a);
        CHECK_CUDA(cudaMemcpy(h_out, d_out, (size_t)num_threads * sizeof(float), cudaMemcpyDeviceToHost));
        printf("k=%d block_size=%d Variant A (contiguous) ms: %.3f +/- %.3f\n",
               k, block_size, mean_a, std_a);

        float mean_b = 0.0f;
        float std_b = 0.0f;
        runKernelStrided(d_in, d_out, n, elems_per_thread, num_threads, block_size, repeats,
                         &mean_b, &std_b);
        CHECK_CUDA(cudaMemcpy(h_out, d_out, (size_t)num_threads * sizeof(float), cudaMemcpyDeviceToHost));
        printf("k=%d block_size=%d Variant B (strided)    ms: %.3f +/- %.3f\n",
               k, block_size, mean_b, std_b);
    }

    // Aufraeumen.
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_in));
    free(h_out);
    free(h_in);

    return 0;
}
