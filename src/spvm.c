#include<stdio.h>
#include<stdlib.h>

#include "loadMatrix.h"

int main(int argc, char* argv[]){

    matrix * m;

    m = loadMatrix(argv[1]);

    printf("Printing matrix\n");
    printMatrix(m);
    
    return 0;
}