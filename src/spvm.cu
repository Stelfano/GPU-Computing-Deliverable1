#include<stdio.h>
#include<stdlib.h>

extern "C" {
    #include "helper.h"
    #include "loadMatrix.h"
}
#include "function.cuh"
#include "cuda_runtime.h"
#include <cusparse.h>

#define TOLERANCE 1e-2

int main(int argc, char* argv[]){

    matrix * m;
    struct timeval start, end;
    float diffCPU, diffGPU;

    m = loadMatrix(argv[1]);
    dtype* ref = generateRandomVector(m->nCols, 10);

    //-------------COO--------------------

    gettimeofday(&start, NULL);
    dtype* res = CPUspvm(m, ref);
    gettimeofday(&end, NULL);
    diffCPU = ((end.tv_sec - start.tv_sec) * 1000.0) + ((end.tv_usec - start.tv_usec) / 1000.0);


    //------------CSR----------------------
/*
    CSRMatrix* csr = cooToCSR(m);

    gettimeofday(&start, NULL);
    dtype* resCSR = CPUspvmCSR(csr, ref);
    gettimeofday(&end, NULL);
    printf("CPU Spmv CSR time: %f ms\n", ((end.tv_sec - start.tv_sec) * 1000.0) + ((end.tv_usec - start.tv_usec) / 1000.0));
*/

    //----------------GPU-COO-Scalar---------------------
    printExperimentBanner("GPU COO-Scalar SpMV", 256, (m->nnz + 255) / 256, m->nnz);

    dtype *Gpuref, *GPURes, *GPUvalues;
    int *GPUrows, *GPUcols;
    int nnz = m->nnz;
    dtype *resAux = (dtype*)malloc(m->nRows * sizeof(dtype));

    printf("Number of non-zero elements: %d\n", nnz);

    copyMatrixGPU(m, &GPUvalues, &GPUrows, &GPUcols, nnz);
    cudaMalloc(&Gpuref, m->nCols * sizeof(dtype));
    cudaMemcpy(Gpuref, ref, m->nCols * sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMalloc(&GPURes, m->nRows * sizeof(dtype));
    cudaMemset(GPURes, 0, m->nRows * sizeof(dtype));

    gettimeofday(&start, NULL);

    //Maybe this shoud go in ad macro at some point
    int num_blocks = (nnz + 255) / 256;
    int threads_per_block = 256;
    spmv_coo_scalar<<<num_blocks, threads_per_block>>>(GPUvalues, GPUrows, GPUcols, nnz, GPURes, Gpuref);
    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) {
        fprintf(stderr, "Error during COO-Scalar kernel execution: %s\n", cudaGetErrorString(err));
        return -1;
    }

    if(cudaMemcpy(resAux, GPURes, m->nRows * sizeof(dtype), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Error copying result back to host: %s\n", cudaGetErrorString(cudaGetLastError()));
        return -1;
    }

    gettimeofday(&end, NULL);
    cudaDeviceSynchronize();

    diffGPU = ((end.tv_sec - start.tv_sec) * 1000.0) + ((end.tv_usec - start.tv_usec) / 1000.0);

    printf("COO-Scalar time: %f ms\n", diffGPU);
    printf("CPU COO time: %f ms\n", diffCPU);
    if(compareVectors(res, resAux, m->nRows, TOLERANCE) == 1){
        printf("Results are approximately equal within the tolerance.\n");
    } else {
        printf("Results differ beyond the tolerance.\n");
    }


    freeCooMatrixGPU(GPUvalues, GPUrows, GPUcols);
    cudaFree(Gpuref);
    cudaFree(GPURes);

    //--------------CSR-SCALAR------------------

    printExperimentBanner("GPU CSR-Scalar SpMV", 256, (m->nnz + 255) / 256, m->nnz);
    printf("Number of non-zero elements: %d\n", nnz);

    CSRMatrix* csr = cooToCSR(m);

    copyCSRMatrixGPU(csr, &GPUvalues, &GPUrows, &GPUcols, nnz);
    cudaMalloc(&Gpuref, csr->nCols * sizeof(dtype));
    cudaMemcpy(Gpuref, ref, csr->nCols * sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMalloc(&GPURes, csr->nRows * sizeof(dtype));
    cudaMemset(GPURes, 0, csr->nRows * sizeof(dtype));

    gettimeofday(&start, NULL);

    //Maybe this shoud go in ad macro at some point
    spmv_csr_scalar<<<num_blocks, threads_per_block>>>(GPUvalues, GPUrows, GPUcols, nnz, GPURes, Gpuref, csr->row_ptr_size);
    err = cudaDeviceSynchronize();
    if(err != cudaSuccess) {
        fprintf(stderr, "Error during CSR-scalar kernel execution: %s\n", cudaGetErrorString(err));
        return -1;
    }

    if(cudaMemcpy(resAux, GPURes, m->nRows * sizeof(dtype), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Error copying result back to host: %s\n", cudaGetErrorString(cudaGetLastError()));
        return -1;
    }

    gettimeofday(&end, NULL);
    cudaDeviceSynchronize();

    diffGPU = ((end.tv_sec - start.tv_sec) * 1000.0) + ((end.tv_usec - start.tv_usec) / 1000.0);

    printf("CSR-Scalar time: %f ms\n", diffGPU);
    printf("CPU COO time: %f ms\n", diffCPU);
    if(compareVectors(res, resAux, csr->nRows, TOLERANCE) == 1){
        printf("Results are approximately equal within the tolerance.\n");
    } else {
        printf("Results differ beyond the tolerance.\n");
    }

    freeCSRMatrixGPU(GPUvalues, GPUrows, GPUcols);
    cudaFree(Gpuref);
    cudaFree(GPURes);


    //---------------- Ell---------------------

    EllpackMatrix* ell = cooToEllpack(m);
    printExperimentBanner("GPU ELL SpMV", 256, (m->nnz + 255) / 256, m->nnz);
    printf("Number of non-zero elements: %d\n", nnz);
    copyEllpackGPU(ell, &GPUvalues, &GPUcols, ell->maxRow, ell->nRows);
    cudaMalloc(&Gpuref, csr->nCols * sizeof(dtype));
    cudaMemcpy(Gpuref, ref, csr->nCols * sizeof(dtype), cudaMemcpyHostToDevice);
    cudaMalloc(&GPURes, ell->nRows * sizeof(dtype));
    cudaMemset(GPURes, 0, ell->nRows * sizeof(dtype));

    gettimeofday(&start, NULL);

    //Maybe this shoud go in ad macro at some point
    spmv_ell<<<num_blocks, threads_per_block>>>(GPUvalues, GPUcols, ell->nRows, ell->maxRow, GPURes, Gpuref);
    err = cudaDeviceSynchronize();
    if(err != cudaSuccess) {
        fprintf(stderr, "Error during ELL-scalar kernel execution: %s\n", cudaGetErrorString(err));
        return -1;
    }

    if(cudaMemcpy(resAux, GPURes, ell->nRows * sizeof(dtype), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fprintf(stderr, "Error copying result back to host: %s\n", cudaGetErrorString(cudaGetLastError()));
        return -1;
    }

    gettimeofday(&end, NULL);
    cudaDeviceSynchronize();

    diffGPU = ((end.tv_sec - start.tv_sec) * 1000.0) + ((end.tv_usec - start.tv_usec) / 1000.0);

    
    //dtype localSum = 0;
    //dtype refSum = 0;
    //for(int i = 0;i<ell->nRows;i++){
    //    localSum += resAux[i];
    //    refSum += res[i];
    //}

    //printf("LS: %f\n", localSum);
    //printf("RS: %f\n", refSum);

    printf("ELL-Scalar time: %f ms\n", diffGPU);
    printf("CPU COO time: %f ms\n", diffCPU);
    if(compareVectors(res, resAux, csr->nRows, TOLERANCE) == 1){
        printf("Results are approximately equal within the tolerance.\n");
    } else {
        printf("Results differ beyond the tolerance.\n");
    }
    


    //--------------CUSparse-COO---------------

   // cuSparseHandle_t cuHandle;

/*
    cuSparseCreate(&cuHandle);

    cuSparseSpMatDescr_t matrixDesc;
    cuSparseDnVecDescr_t vecRef, vecRes;
    size_t bufferSize;
    void* dBuffer;
    float *alpha, *beta;
    alpha->1.0f;
    beta->0.0f;

    cusparseCreateCOO(&matrixDesc, m->nRows, m->nCols, m->nnz, m->rows, m->cols, m->data, 
                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32);
    cusparseCreateDnVec(&vecRef, m->nCols, &Gpuref, CUDA_R_32F);
    cusparseCreateDnVec(&vecRes, m->nRows, &GPURes, CUDA_R_32F);

    cusparseSpMV_bufferSize(cuHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matrixDesc, 
                            vecRef, &beta, vecRes, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    cusparseSpMV(cuHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matrixDesc, vecRef, &beta, vecRes,
                 CUDA_R_32F, cusparse_SPMV_ALG_DEFAULT, dBuffer);

    cudaMemcpy(resAux, GPURes, m->nRows * sizeof(dtype), cudaMemcpyDeviceToHost);

    if(compareVectors(res, resAux, m->nRows, TOLERANCE) == 1){
        printf("Results are approximately equal within the tolerance.\n");
    } else {
        printf("Results differ beyond the tolerance.\n");
    }

    cuSparseDestorySpMat(matrixDesc);
    cuSparseDestroyDnVec(vecRef);
    cuSparseDestroyDnVec(vecRes);
    cuSparseDestroy(cuHandle);

*/
    //------------- FREE MEMORY ----------------
    free(ref);
    free(res);
    freeCSR(csr);
    //freeMatrix(m);
   // freeEllpack(ell);
    free(resAux);

    
    return 0;
}