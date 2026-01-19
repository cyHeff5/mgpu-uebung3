#ifndef TASK3_MATRIX_H
#define TASK3_MATRIX_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Host-Hilfsfunktionen fuer Matrix und Vektor im Vollformat.
float* matrix_alloc(size_t n);
float* vector_alloc(size_t n);
void matrix_init_upper(float* a, size_t n);
void vector_init(float* x, size_t n);
void matrix_free(float* a);
void vector_free(float* x);

#ifdef __cplusplus
} 
#endif

#endif  // TASK3_MATRIX_H
