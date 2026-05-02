#include "function.cuh"
#include "stdio.h"
#include <cuda_runtime.h>

__global__ void spmv_coo_kernel_ptr(const matrix* A, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    printf("Thread %d processing element %d\n", threadIdx.x, i);
    // Accediamo ai campi con l'operatore freccia ->
    if (i < A->nnz) {
        int row = A->rows[i];
        int col = A->cols[i];
        double val = A->data[i];

        atomicAdd(&y[row], val * x[col]);
    }
}

matrix* copyMatrixGPU(const matrix *m) {
    matrix* mat; // Copia locale della struct (su CPU)

    cudaMallocManaged(&mat, sizeof(matrix));
    cudaMallocManaged(&mat->rows, m->nnz * sizeof(int));
    cudaMallocManaged(&mat->cols, m->nnz * sizeof(int));
    cudaMallocManaged(&mat->data, m->nnz * sizeof(double));

    mat->nnz = m->nnz;
    mat->nRows = m->nRows;
    mat->nCols = m->nCols;

    return mat;
}

void freeCooMatrix(matrix* m) {
    cudaFree(m->rows);
    cudaFree(m->cols);
    cudaFree(m->data);
    cudaFree(m);
}