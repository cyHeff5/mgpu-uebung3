#include "matrix.h"

#include <stdlib.h>

float* matrix_alloc(size_t n) {
    // Volle NxN-Matrix (einfachstes Layout).
    return (float*)malloc(n * n * sizeof(float));
}

float* vector_alloc(size_t n) {
    return (float*)malloc(n * sizeof(float));
}

void matrix_init_upper(float* a, size_t n) {
    // Untere Dreieckshaelfte auf 0, Rest zufaellig.
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (j < i) {
                a[i * n + j] = 0.0f;
            } else {
                a[i * n + j] = (float)rand() / (float)RAND_MAX;
            }
        }
    }
}

void vector_init(float* x, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        x[i] = (float)rand() / (float)RAND_MAX;
    }
}

void matrix_free(float* a) {
    free(a);
}

void vector_free(float* x) {
    free(x);
}
