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

    float* ref = generateRandomVector(m->nRows, 10);
    gettimeofday(&start, NULL);
    float* res = CPUspvmParallel(m, ref);
    gettimeofday(&end, NULL);
    printf("CPU Spmv time: %f ms\n", ((end.tv_sec - start.tv_sec) * 1000.0) + ((end.tv_usec - start.tv_usec) / 1000.0));

    printf("\nFinal res:\n");
    for(int i=0;i<m->nRows;i++){
        printf("%d : %f\n",i , res[i]);
    }

    printf("Convert to CSR format\n");
    CSRMatrix* csr = cooToCSR(m);
    printCSR(csr);

    float* resCSR = CPUspvmCSR(csr, ref);
    printf("\nFinal res with CSR:\n");
    for(int i=0;i<csr->nCols;i++){
        printf("%d : %f\n",i , resCSR[i]);
    }

    matrix *mat = copyMatrixGPU(m);
    float* GpuVector = (float*)malloc(m->nCols * sizeof(float));
    float* gpuRes = (float*)malloc(m->nRows * sizeof(float));

    cudaMallocManaged(&GpuVector, m->nCols * sizeof(float));
    cudaMallocManaged(&gpuRes, m->nRows * sizeof(float));
    //calculate time for GPU Spmv


    gettimeofday(&start, NULL);

    //Start the kernel for Spmv
    int blockSize = 256;
    int numBlocks = (m->nnz + blockSize - 1) / blockSize;
    spmv_coo_kernel_ptr<<<numBlocks, blockSize>>>(mat, GpuVector, gpuRes);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    printf("GPU Spmv time: %f ms\n", ((end.tv_sec - start.tv_sec) * 1000.0) + ((end.tv_usec - start.tv_usec) / 1000.0));
    printf("\nFinal res with GPU:\n");
    for(int i=0;i<m->nRows;i++){
        printf("%d : %f\n",i , gpuRes[i]);
    }
    if(compareVectors(gpuRes, resCSR, m->nRows, TOLERANCE)){
        printf("\nResults are approximately equal within the tolerance.\n");
    } else {
        printf("\nResults differ beyond the tolerance.\n");
    }


    freeCooMatrix(mat);

    freeMatrix(m);
    free(ref);
    free(res);

    freeCSR(csr);
    free(resCSR);
    
    return 0;
}