#ifndef TASK3_KERNELS_CUH
#define TASK3_KERNELS_CUH

// Blockweise MatVec fuer obere Dreiecksmatrix.
__global__ void matvec_upper(const float* a, const float* x, float* y, int n);

#endif  // TASK3_KERNELS_CUH
