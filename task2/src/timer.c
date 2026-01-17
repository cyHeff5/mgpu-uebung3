#include "timer.h"

#include <omp.h>

double now_seconds(void) {
    return omp_get_wtime();
}
