#include "matvec.h"

#include <omp.h>

void matvec_seq(const float* a, const float* x, float* y, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        // Pro Zeile nur ab der Diagonale rechnen.
        float sum = 0.0f;
        size_t row = i * n;
        for (size_t j = i; j < n; ++j) {
            sum += a[row + j] * x[j];
        }
        y[i] = sum;
    }
}

void matvec_omp(const float* a, const float* x, float* y, size_t n, int num_threads) {
    // Guided Scheduling gleicht ungleiche Zeilenlaengen aus.
    #pragma omp parallel for schedule(guided) num_threads(num_threads)
    for (long long i = 0; i < (long long)n; ++i) {
        float sum = 0.0f;
        size_t row = (size_t)i * n;
        for (size_t j = (size_t)i; j < n; ++j) {
            sum += a[row + j] * x[j];
        }
        y[i] = sum;
    }
}
