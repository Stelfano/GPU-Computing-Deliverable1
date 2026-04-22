//Helper functions for matrix operations

struct matrix {
    int nnz;
    int* rows;
    int* cols;
    double* data;
}typedef matrix;

void printMatrix(matrix* m);
void freeMatrix(matrix* m);