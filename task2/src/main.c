#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "matrix.h"
#include "matvec.h"
#include "timer.h"

static void usage(const char* prog) {
    // Einfache CLI: N und optional Threadzahl.
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

    // Jede Messung mehrfach ausfuehren, damit der Mittelwert stabiler ist.
    const int repeats = 5;

    // Speicher fuer Matrix und Vektoren anlegen.
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

    // Zufallsdaten: obere Dreiecksmatrix und Vektor.
    matrix_init_upper(a, n);
    vector_init(x, n);

    // Sequentiell und OpenMP jeweils mehrfach messen und mitteln.
    double seq_mean = 0.0;
    double seq_m2 = 0.0;
    double omp_mean = 0.0;
    double omp_m2 = 0.0;
    for (int r = 0; r < repeats; ++r) {
        double t0 = now_seconds();
        matvec_seq(a, x, y_seq, n);
        double t1 = now_seconds();
        double seq_s = t1 - t0;
        double seq_delta = seq_s - seq_mean;
        seq_mean += seq_delta / (double)(r + 1);
        double seq_delta2 = seq_s - seq_mean;
        seq_m2 += seq_delta * seq_delta2;

        double t2 = now_seconds();
        matvec_omp(a, x, y_omp, n, num_threads);
        double t3 = now_seconds();
        double omp_s = t3 - t2;
        double omp_delta = omp_s - omp_mean;
        omp_mean += omp_delta / (double)(r + 1);
        double omp_delta2 = omp_s - omp_mean;
        omp_m2 += omp_delta * omp_delta2;
    }

    double seq_s = seq_mean;
    double omp_s = omp_mean;
    double seq_std = (repeats > 1) ? sqrt(seq_m2 / (double)(repeats - 1)) : 0.0;
    double omp_std = (repeats > 1) ? sqrt(omp_m2 / (double)(repeats - 1)) : 0.0;
    double speedup = seq_s / omp_s;

    printf("N=%zu threads=%d\n", n, num_threads);
    printf("Sequential: %.6f s +/- %.6f\n", seq_s, seq_std);
    printf("OpenMP:     %.6f s +/- %.6f\n", omp_s, omp_std);
    printf("Speedup:    %.3f\n", speedup);

    matrix_free(a);
    vector_free(x);
    vector_free(y_seq);
    vector_free(y_omp);
    return 0;
}
