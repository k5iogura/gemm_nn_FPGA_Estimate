#include <stdio.h>
#include <stdlib.h>
#include "gemm_fpga.h"
int main(){
  float *A=(float*)malloc(1000*sizeof(float));
  float *B=(float*)malloc(1000*sizeof(float));
  float *C=(float*)malloc(1000*sizeof(float));
  gemm_fpga_init();
  gemm_nn_fpga(1,1,1,1.f,A,0,B,0,C,0);
  gemm_fpga_finalize();
}

