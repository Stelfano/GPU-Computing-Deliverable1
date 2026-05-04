
//Helper functions for matrix operations and more
#ifndef HELPER_H
#define HELPER_H
#include <sys/time.h>
#define dtype float

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


dtype* generateRandomVector(int size, int maxVal);
dtype* CPUspvm(matrix *m, dtype* vector);
dtype* CPUspvmParallel(matrix *m, dtype* vector);

CSRMatrix* cooToCSR(matrix* m);
void printCSR(CSRMatrix *m);
void freeCSR(CSRMatrix *m);
dtype *CPUspvmCSR(CSRMatrix *m, dtype* vector);
dtype *CPUspvmParallelCSR(CSRMatrix *m, dtype* vector);

int compareVectors(dtype* a, dtype* b, int size, dtype tolerance);

#endif