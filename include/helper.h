
//Helper functions for matrix operations and more
#ifndef HELPER_H
#define HELPER_H
#include <sys/time.h>

#define TIMER_DEF     struct timeval temp_1, temp_2
#define TIMER_START   gettimeofday(&temp_1, (struct timezone*)0)
#define TIMER_STOP    gettimeofday(&temp_2, (struct timezone*)0)
#define TIMER_ELAPSED ((temp_2.tv_sec-temp_1.tv_sec)*1000.0 +(temp_2.tv_usec-temp_1.tv_usec)/1000.0) // return ms

typedef struct {
    int nnz;
    int nRows;
    int nCols;
    int* rows;
    int* cols;
    float* data;
} matrix;

typedef struct {
    int nRows;
    int nCols;
    int nnz;
    int *row_ptr;    // Dimensione: rows + 1
    int *col_indices; // Dimensione: nnz
    float *data;   // Dimensione: nnz
} CSRMatrix;

void printMatrix(matrix* m);
void freeMatrix(matrix* m);

float* generateRandomVector(int size, int maxVal);
float* CPUspvm(matrix *m, float* vector);
float* CPUspvmParallel(matrix *m, float* vector);

CSRMatrix* cooToCSR(matrix* m);
void printCSR(CSRMatrix *m);
void freeCSR(CSRMatrix *m);
float *CPUspvmCSR(CSRMatrix *m, float* vector);
float *CPUspvmParallelCSR(CSRMatrix *m, float* vector);

int compareVectors(float* a, float* b, int size, float tolerance);

#endif