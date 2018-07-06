#ifdef CTEST
#include <stdio.h>
#include <malloc.h>
#endif

#define W (224)
#define H (160)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

float DOT(float A[9], float B[9]){
    return 
    A[0] * B[0] +
    A[1] * B[1] +
    A[2] * B[2] +
    A[3] * B[3] +
    A[4] * B[4] +
    A[5] * B[5] +
    A[6] * B[6] +
    A[7] * B[7] +
    A[8] * B[8] ;
}
#ifdef CTEST
#define MCLASS float *
#else
#define MCLASS global float *restrict
kernel
#endif
void SRConv (const int M, const int N, const int K, const float ALPHA,
		 MCLASS A, const int lda,
		 MCLASS B, const int ldb,
		 MCLASS C, const int ldc
		)
{
    int i,j,k;
    float SR[W*2+3];
    float krn[9] = { A[    0],  A[    1],  A[    2],
                     A[    3],  A[    4],  A[    5],
                     A[    6],  A[    7],  A[    8]};
    //for(j=0;j<2*W+3;j++)
        //SR[j] = 0;
    for(i=0,k=0;i<W*H;i++){
        #pragma unroll
        for(j=2*W+3-1;j>0;j--)
            SR[j] = SR[j-1];
        SR[0] = B[i];
        float pch[9] = {SR[    0], SR[    1], SR[    2],
                        SR[  W+0], SR[  W+1], SR[  W+2],
                        SR[2*W+0], SR[2*W+1], SR[2*W+2]};
        unsigned int v = (i >= 2*W+3) && (( i - (2*W+3) ) % W <= W-3);
        C[k]+= DOT(krn, pch) * (float)v;
        k   += v;
    }
}

#ifdef CTEST
int main(){
    int i;
    int S=3;    //KERNEL SIZE
    int I=3;    //IN CHANNELS
    int M=8;    //GO CHANNELS
    int N=W*H, K=S*S*I;
    float *A=(float*)malloc(sizeof(float)*M*K);
    float *B=(float*)malloc(sizeof(float)*K*N);
    float *C=(float*)malloc(sizeof(float)*M*N);

    A[0]= 2; A[1]= 1; A[2]= 1;
    A[3]= 1; A[4]= 1; A[5]= 1;
    A[6]= 1; A[7]= 1; A[8]= 2;
    for(i=0;i<W*H;i++) B[i]=.1;
    SRConv(M,N,K,1,A,K,B,N,C,N);
}
#endif

