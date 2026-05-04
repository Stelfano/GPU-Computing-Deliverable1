#include<stdio.h>
#include<stdlib.h>

extern "C" {
    #include "helper.h"
    #include "loadMatrix.h"
}
#include "function.cuh"
#include "cuda_runtime.h"

#define TOLERANCE 1e-5

int main(int argc, char* argv[]){

    matrix * m;
    struct timeval start, end;

    m = loadMatrix(argv[1]);

    printf("Printing matrix\n");
    printMatrix(m);

    dtype* ref = generateRandomVector(m->nRows, 10);

    //-------------COO--------------------

    gettimeofday(&start, NULL);
    dtype* res = CPUspvm(m, ref);
    gettimeofday(&end, NULL);
    printf("CPU Spmv time: %f ms\n", ((end.tv_sec - start.tv_sec) * 1000.0) + ((end.tv_usec - start.tv_usec) / 1000.0));

    //------------CSR----------------------

    CSRMatrix* csr = cooToCSR(m);
    dtype* resCSR = CPUspvmCSR(csr, ref);

    //----------------GPU-COO---------------------
    dtype *Gpuref, *GPURes, *GPUvalues;
    int *GPUrows, *GPUcols;
    int nnz = m->nnz;

    printf("Number of non-zero elements: %d\n", nnz);

    copyMatrixGPU(m, GPUvalues, GPUrows, GPUcols, nnz);

    cudaMalloc(&Gpuref, m->nRows * sizeof(dtype));
    cudaMemcpy(Gpuref, ref, m->nRows * sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMalloc(&GPURes, m->nRows * sizeof(dtype));

    gettimeofday(&start, NULL);
    
    //Maybe this shoud go in ad macro at some point
    int num_blocks = (nnz + 255) / 256;
    int threads_per_block = 256;

    spmv_coo_scalar<<<num_blocks, threads_per_block>>>(GPUvalues, GPUrows, GPUcols, nnz, GPURes, Gpuref);
    cudaDeviceSynchronize();

    cudaMemcpy(res, GPURes, m->nRows * sizeof(dtype), cudaMemcpyDeviceToHost);
    gettimeofday(&end, NULL);

    printf("GPU Spmv time: %f ms\n", ((end.tv_sec - start.tv_sec) * 1000.0) + ((end.tv_usec - start.tv_usec) / 1000.0));
    printf("\nFinal result between GPU:\n");

    for(int i = 0;i<m->nRows; i++){
        printf("%d = %f\n", i, fabs(res[i]));
    }

    if(compareVectors(res, resCSR, m->nRows, TOLERANCE) == 1){
        printf("\nResults are approximately equal within the tolerance.\n");
    } else {
        printf("\nResults differ beyond the tolerance.\n");
    }

    //------------- FREE MEMORY ----------------
    free(ref);
    free(res);

    freeCSR(csr);
    free(resCSR);

    cudaFree(Gpuref);
    cudaFree(GPURes);
    freeCooMatrixGPU(GPUvalues, GPUrows, GPUcols);
    
    return 0;
}