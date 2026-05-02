#ifndef FUNCTION_CUH
#define FUNCTION_CUH
#include "helper.h"
#include <cuda_runtime.h>

__global__ void spmv_coo_kernel_ptr(const matrix* A, const float* x, float* y);
matrix* copyMatrixGPU(const matrix *m);
void freeCooMatrix(matrix* m);

#endif