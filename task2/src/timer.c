#include "timer.h"

#include <omp.h>

double now_seconds(void) {
    // OpenMP liefert eine einfache Wandzeit.
    return omp_get_wtime();
}
