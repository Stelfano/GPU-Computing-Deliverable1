
//Helper functions for matrix operations and more
#ifndef HELPER_H
#define HELPER_H
#include <sys/time.h>
#define dtype float
#define XSTR(x) STR(x)
#define STR(x) #x

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
    int row_ptr_size; // Dimensione di row_ptr
} CSRMatrix;

typedef struct {
    int nRows;          // Numero righe
    int maxRow;          // Max elementi per riga
    dtype *values;  
    int *cols;      
} EllpackMatrix;

typedef struct{
    int totalRows;
    int sliceSize;
    int *hackOffset;
    int *nRows;
    int *maxRows;
    int *cols;
    dtype *values;
}HellpackMatrix;

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

EllpackMatrix* cooToEllpack(matrix *m);
void freeEllpack(EllpackMatrix* ell);
void printEllpack(EllpackMatrix *ell);

int compareVectors(dtype* a, dtype* b, int size, dtype tolerance);
void printExperimentBanner(const char* title, int numThreads, int numBlocks, int nnz);

#endif