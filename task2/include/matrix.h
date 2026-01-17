#ifndef TASK2_MATRIX_H
#define TASK2_MATRIX_H

#include <stddef.h>

// Hilfsfunktionen fuer Matrix/Vektor im Vollformat.

float* matrix_alloc(size_t n);
float* vector_alloc(size_t n);
void matrix_init_upper(float* a, size_t n);
void vector_init(float* x, size_t n);
void matrix_free(float* a);
void vector_free(float* x);

#endif  // TASK2_MATRIX_H
