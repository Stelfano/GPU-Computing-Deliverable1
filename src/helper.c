#include<stdio.h>
#include<stdlib.h>

#include "helper.h"

void printMatrix(matrix* m){
    for(int i = 0; i < m->nnz; i++){
        printf("(%d, %d) = %lf\n", m->rows[i], m->cols[i], m->data[i]);
    }
}

void freeMatrix(matrix* m){
    free(m->rows);
    free(m->cols);
    free(m->data);
    free(m);
}