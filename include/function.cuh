#ifndef FUNCTION_CUH
#define FUNCTION_CUH
#include "helper.h"
#include <cuda_runtime.h>

__global__ void sample();
__global__ void spmv_coo_scalar(const dtype* GPUvalues, const int *GPUrows, const int *GPUcols, int nnz, dtype * res, dtype* ref);
void copyMatrixGPU(const matrix *m, dtype** GPUvalues, int** GPUrows, int** GPUcols, int nnz);
void freeCooMatrixGPU(dtype* GPUvalues, int* GPUrows, int* GPUcols);

__global__ void spmv_csr_scalar(const dtype* GPUvalues, const int* GPUrowPtr, const int *Gpucols, int nnz, dtype *res, dtype *ref, const int RowPtrSize);
void copyCSRMatrixGPU(const CSRMatrix *m, dtype **GPUvalues, int** GPUrows, int** GPUcols, int nnz);
void freeCSRMatrixGPU(dtype *GPUvalues, int* GPUrows, int* GPUcols);

//--------------------GPU-ELLPACK--------------------------
void copyEllpackGPU(const EllpackMatrix* ell, dtype** GPUvalues, int **GPUcols, int maxRow, int nRows);
__global__ void spmv_ell(const dtype* GPUvalues, const int* GPUcols, const int nRows, const int maxRow, dtype* res, const dtype* ref);

#endif