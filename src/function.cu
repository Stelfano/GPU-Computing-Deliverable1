#include "function.cuh"
#include <stdio.h>
#include <cuda_runtime.h>

//--------------------GPU-COO-Scalar---------------------

__global__ void spmv_coo_scalar(const dtype* GPUvalues, const int *GPUrows, const int *GPUcols, int nnz, dtype * res, dtype *ref) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x*blockDim.x;

    for (int idx = i; idx < nnz; idx += totalThreads) {
        atomicAdd(&res[GPUrows[idx]], GPUvalues[idx] * ref[GPUcols[idx]]);
    }
}

void copyMatrixGPU(const matrix *m, dtype** GPUvalues, int** GPUrows, int** GPUcols, int nnz) {

    cudaMalloc(GPUvalues, nnz * sizeof(dtype));
    cudaMalloc(GPUrows, nnz * sizeof(int));
    cudaMalloc(GPUcols, nnz * sizeof(int));

    cudaMemcpy(*GPUvalues, m->data, nnz*sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(*GPUrows, m->rows, nnz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(*GPUcols, m->cols, nnz*sizeof(int), cudaMemcpyHostToDevice);
}

void freeCooMatrixGPU(dtype* GPUvalues, int* GPUrows, int* GPUcols){
    cudaFree(GPUvalues);
    cudaFree(GPUrows);
    cudaFree(GPUcols);
}


//--------------------GPU-CSR-Scalar---------------------
void copyCSRMatrixGPU(const CSRMatrix *m, dtype** GPUvalues, int** GPUrows, int** GPUcols, int nnz){
    cudaMalloc(GPUvalues, nnz * sizeof(dtype));
    cudaMalloc(GPUrows, nnz * sizeof(int));
    cudaMalloc(GPUcols, nnz * sizeof(int));

    cudaMemcpy(*GPUvalues, m->data, nnz*sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(*GPUrows, m->row_ptr, m->row_ptr_size*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(*GPUcols, m->col_indices, nnz*sizeof(int), cudaMemcpyHostToDevice);
}
__global__ void spmv_csr_scalar(const dtype* GPUvalues, const int* GPUrowPtr, const int *GPUcols, int nnz, dtype *res, dtype *ref, const int RowPtrSize){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx <= RowPtrSize-1){
        int row_start = GPUrowPtr[idx];
        int row_end;
        if(idx != RowPtrSize-1) //Questo if da solo mi ha fatto perdere 2ms su Webbase
            row_end = GPUrowPtr[idx+1];
        else
            row_end = nnz;

        for(int i = 0;i < row_end - row_start; i++){
            res[idx] += GPUvalues[row_start + i] * ref[GPUcols[row_start + i]];
        }
    }
}

void freeCSRMatrixGPU(dtype* GPUvalues, int* GPUrows, int* GPUcols){
    cudaFree(GPUvalues);
    cudaFree(GPUrows);
    cudaFree(GPUcols);
}