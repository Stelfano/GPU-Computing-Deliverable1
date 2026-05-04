#ifndef FUNCTION_CUH
#define FUNCTION_CUH
#include "helper.h"
#include <cuda_runtime.h>

__global__ void spmv_coo_scalar(const dtype* GPUvalues, const int *GPUrows, const int *GPUcols, int nnz, dtype * res, dtype* ref);
void copyMatrixGPU(const matrix *m, dtype* GPUvalues, int* GPUrows, int* GPUcols, int nnz);
void freeCooMatrixGPU(dtype* GPUvalues, int* GPUrows, int* GPUcols);

#endif