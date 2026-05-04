#include "function.cuh"
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void spmv_coo_scalar(const dtype* GPUvalues, const int *GPUrows, const int *GPUcols, int nnz, dtype * res, dtype *ref) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x*blockDim.x;

    for (int idx = i; idx < nnz; idx += totalThreads) {
        atomicAdd(&res[GPUrows[idx]], GPUvalues[idx] * ref[GPUcols[idx]]);
    }

    printf("Thread %d finished processing.\n", i);
}

void copyMatrixGPU(const matrix *m, dtype* GPUvalues, int* GPUrows, int* GPUcols, int nnz) {

    cudaMalloc(&GPUvalues, nnz * sizeof(dtype));
    cudaMalloc(&GPUrows, nnz * sizeof(int));
    cudaMalloc(&GPUcols, nnz * sizeof(int));

    cudaMemcpy(GPUvalues, m->data, nnz*sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMemcpy(GPUrows, m->rows, nnz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(GPUcols, m->cols, nnz*sizeof(int), cudaMemcpyHostToDevice);

    printf("Matrix copied to GPU");
}

void freeCooMatrixGPU(dtype* GPUvalues, int* GPUrows, int* GPUcols){
    cudaFree(GPUvalues);
    cudaFree(GPUrows);
    cudaFree(GPUcols);
}