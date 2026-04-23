#include<stdio.h>
#include<stdlib.h>
#include<omp.h>

#include "helper.h"

void printMatrix(matrix* m){
    printf("The matrix is of size: %d, %d\n",m->nRows, m->nCols);

    for(int i = 0; i < m->nnz; i++){
        printf("(%d, %d) = %f\n", m->rows[i], m->cols[i], m->data[i]);
    }
}

void freeMatrix(matrix* m){
    free(m->rows);
    free(m->cols);
    free(m->data);
    free(m);
}

float* generateRandomVector(int size, int maxVal){
    srand(42);
    float* vector = (float*)malloc(size * sizeof(float));


    for (int i = 0; i < size; i++) {
        vector[i] = ((float)rand() / (float)RAND_MAX) * maxVal; // Generate random float between 0 and 1
    }

    return vector;
}

float* CPUspvm(matrix *m, float* vector){
    float* y = (float *)malloc(sizeof(float) * m->nRows);

    for (int i = 0; i < m->nRows; i++) {
        y[i] = 0.0;
    }

    for (int i = 0; i < m->nnz; i++) {
        int r = m->rows[i];
        int c = m->cols[i];
        float val = m->data[i];
        y[r] += val * vector[c];
    }

    return y;
}

float* CPUspvmParallel(matrix *m, float* vector){
    omp_set_num_threads(4);

    printf("\nExecuting CPU Spvm with openmp threads...\n", omp_get_num_threads());
    float* y = (float *)malloc(sizeof(float) * m->nRows);

    #pragma omp parallel for
    for (int i = 0; i < m->nRows; i++) {
        y[i] = 0.0;
    }

    #pragma omp parallel for
    for (int i = 0; i < m->nnz; i++) {
        int r = m->rows[i];
        int c = m->cols[i];
        float val = m->data[i];

        #pragma omp atomic
        y[r] += val * vector[c];
    }

    return y;
}