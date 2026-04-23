#include<stdio.h>
#include<stdlib.h>

#include "loadMatrix.h"

int main(int argc, char* argv[]){

    matrix * m;

    m = loadMatrix(argv[1]);

    printf("Printing matrix\n");
    printMatrix(m);

    float* ref = generateRandomVector(m->nRows, 10);
    float* res = CPUspvmParallel(m, ref);

    printf("\nFinal res:\n");
    for(int i=0;i<m->nRows;i++){
        printf("%d : %f\n",i , res[i]);
    }

    freeMatrix(m);
    free(ref);
    free(res);
    
    return 0;
}