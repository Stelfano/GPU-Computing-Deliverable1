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

    printf("Convert to CSR format\n");
    CSRMatrix* csr = cooToCSR(m);
    printCSR(csr);

    float* resCSR = CPUspvmParallelCSR(csr, ref);
    printf("\nFinal res with CSR:\n");
    for(int i=0;i<csr->nCols;i++){
        printf("%d : %f\n",i , resCSR[i]);
    }

    if(compareVectors(res, resCSR, m->nRows, 1e-5)){
        printf("\nResults are approximately equal within the tolerance.\n");
    } else {
        printf("\nResults differ beyond the tolerance.\n");
    }

    freeMatrix(m);
    free(ref);
    free(res);

    freeCSR(csr);
    free(resCSR);
    
    return 0;
}