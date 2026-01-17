#include <stdio.h>
#include <stdlib.h>

static int try_alloc(size_t n) {
    // Ueberlaufpruefungen fuer n*n und Bytegroesse.
    if (n > SIZE_MAX / n) {
        return 0;
    }
    size_t elems = n * n;
    if (elems > SIZE_MAX - 3 * n) {
        return 0;
    }
    elems += 3 * n;
    if (elems > SIZE_MAX / sizeof(float)) {
        return 0;
    }
    float* buf = (float*)malloc(elems * sizeof(float));
    if (!buf) {
        return 0;
    }
    // Erste und letzte Stelle anfassen, damit der Speicher committed wird.
    buf[0] = 0.0f;
    buf[elems - 1] = 1.0f;
    free(buf);
    return 1;
}

int main(void) {
    size_t n = 1024;
    size_t last_ok = 0;

    while (try_alloc(n)) {
        last_ok = n;
        printf("OK: N=%zu\n", n);
        n *= 2;
        if (n == 0) {
            break;
        }
    }

    size_t low = last_ok;
    size_t high = n;
    while (low + 1 < high) {
        size_t mid = low + (high - low) / 2;
        if (try_alloc(mid)) {
            low = mid;
        } else {
            high = mid;
        }
    }

    printf("Max N (approx): %zu\n", low);
    return 0;
}
