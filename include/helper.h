//Helper functions for matrix operations

struct matrix {
    int nnz;
    int nRows;
    int nCols;
    int* rows;
    int* cols;
    float* data;
}typedef matrix;

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