//Helper functions for matrix operations

struct matrix {
    int nnz;
    int nRows;
    int nCols;
    int* rows;
    int* cols;
    float* data;
}typedef matrix;

struct CSRmatrix {
    int nnz;
    int nRows;
    int nCols;
    int* rowPtr;
    int* colInd;
    float* data;
}typedef CSRmatrix;

void printMatrix(matrix* m);
void freeMatrix(matrix* m);

float* generateRandomVector(int size, int maxVal);
float* CPUspvm(matrix *m, float* vector);
float* CPUspvmParallel(matrix *m, float* vector);