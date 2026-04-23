#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
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


CSRMatrix* cooToCSR(matrix *coo) {
    CSRMatrix* csr = (CSRMatrix * )malloc(sizeof(CSRMatrix));
    csr->nRows = coo->nRows;
    csr->nCols = coo->nCols;
    csr->nnz = coo->nnz;

    // 1. Allocazione memoria
    csr->row_ptr = (int *)calloc(csr->nRows + 1, sizeof(int));
    csr->col_indices = (int *)malloc(csr->nnz * sizeof(int));
    csr->data = (float *)malloc(csr->nnz * sizeof(float));

    // 2. Contiamo gli elementi per ogni riga
    for (int i = 0; i < coo->nnz; i++) {
        csr->row_ptr[coo->rows[i]]++;
    }

    // 3. Trasformiamo i conteggi in offset (Somma Prefissa)
    // row_ptr[i] ora indicherà dove inizia la riga i
    int cumulative_sum = 0;
    for (int i = 0; i <= csr->nRows; i++) {
        int temp = csr->row_ptr[i];
        csr->row_ptr[i] = cumulative_sum;
        cumulative_sum += temp;
    }

    // 4. Copiamo i dati nelle posizioni finali
    // Usiamo un vettore temporaneo per sapere dove inserire il prossimo elemento di ogni riga
    int *temp_row_ptr = (int *)malloc(csr->nRows * sizeof(int));
    memcpy(temp_row_ptr, csr->row_ptr, csr->nRows * sizeof(int));

    for (int i = 0; i < coo->nnz; i++) {
        int row = coo->rows[i];
        int dest_pos = temp_row_ptr[row];

        csr->col_indices[dest_pos] = coo->cols[i];
        csr->data[dest_pos] = coo->data[i];

        temp_row_ptr[row]++;
    }

    free(temp_row_ptr);
    return csr;
}

void printCSR(CSRMatrix *m){
    printf("The matrix is of size: %d, %d\n",m->nRows, m->nCols);

    for(int i = 0; i < m->nRows; i++){
        for(int j = m->row_ptr[i]; j < m->row_ptr[i + 1]; j++){
            printf("(%d, %d) = %f\n", i, m->col_indices[j], m->data[j]);
        }
    }
}

void freeCSR(CSRMatrix *m){
    free(m->row_ptr);
    free(m->col_indices);
    free(m->data);
    free(m);
}

float *CPUspvmCSR(CSRMatrix *m, float* vector){
    float* y = (float *)malloc(sizeof(float) * m->nRows);

    for (int i = 0; i < m->nRows; i++) {
        y[i] = 0.0;
    }

    for (int i = 0; i < m->nRows; i++) {
        for (int j = m->row_ptr[i]; j < m->row_ptr[i + 1]; j++) {
            int col = m->col_indices[j];
            float val = m->data[j];
            y[i] += val * vector[col];
        }
    }

    return y;
}

float *CPUspvmParallelCSR(CSRMatrix *m, float* vector){
    omp_set_num_threads(4);

    printf("\nExecuting CPU Spvm with openmp threads...\n", omp_get_num_threads());
    float* y = (float *)malloc(sizeof(float) * m->nRows);

    #pragma omp parallel for
    for (int i = 0; i < m->nRows; i++) {
        y[i] = 0.0;
    }

    #pragma omp parallel for
    for (int i = 0; i < m->nRows; i++) {
        for (int j = m->row_ptr[i]; j < m->row_ptr[i + 1]; j++) {
            int col = m->col_indices[j];
            float val = m->data[j];

            #pragma omp atomic
            y[i] += val * vector[col];
        }
    }

    return y;
}

int compareVectors(float* a, float* b, int size, float tolerance) {
    for (int i = 0; i < size; i++) {
        if (fabs(a[i] - b[i]) > tolerance) {
            return 0; 
        }
    }
    return 1; 
}