#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"
#include "matvec.h"
#include "timer.h"

static void usage(const char* prog) {
    printf("Usage: %s N [num_threads]\n", prog);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    size_t n = (size_t)strtoull(argv[1], NULL, 10);
    if (n == 0) {
        fprintf(stderr, "N must be > 0\n");
        return 1;
    }

    int num_threads = 0;
    if (argc >= 3) {
        num_threads = atoi(argv[2]);
    }
    if (num_threads <= 0) {
        num_threads = 1;
    }

    float* a = matrix_alloc(n);
    float* x = vector_alloc(n);
    float* y_seq = vector_alloc(n);
    float* y_omp = vector_alloc(n);
    if (!a || !x || !y_seq || !y_omp) {
        fprintf(stderr, "Allocation failed\n");
        matrix_free(a);
        vector_free(x);
        vector_free(y_seq);
        vector_free(y_omp);
        return 1;
    }

    matrix_init_upper(a, n);
    vector_init(x, n);

    double t0 = now_seconds();
    matvec_seq(a, x, y_seq, n);
    double t1 = now_seconds();

    double t2 = now_seconds();
    matvec_omp(a, x, y_omp, n, num_threads);
    double t3 = now_seconds();

    double seq_s = t1 - t0;
    double omp_s = t3 - t2;
    double speedup = seq_s / omp_s;

    printf("N=%zu threads=%d\n", n, num_threads);
    printf("Sequential: %.6f s\n", seq_s);
    printf("OpenMP:     %.6f s\n", omp_s);
    printf("Speedup:    %.3f\n", speedup);

    matrix_free(a);
    vector_free(x);
    vector_free(y_seq);
    vector_free(y_omp);
    return 0;
}
