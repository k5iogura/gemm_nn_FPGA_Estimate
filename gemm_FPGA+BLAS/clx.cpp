#define __USE_MINGW_ANSI_STDIO 1
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "fp16.h"
#include <cblas.h>
//#include <OpenEXR/half.h>

#include <CL/cl.h>
#include "cl_body.h"

void init_clock_realmsec(){
	timespec ts;
	ts.tv_sec=0;
	ts.tv_nsec=0;
	clock_settime(CLOCK_REALTIME,&ts);
}
double clock_realmsec(){
	timespec ts;
	clock_gettime(CLOCK_MONOTONIC,&ts);
	return (ts.tv_sec+ts.tv_nsec*1e-9)*1000.;
}

void row2col_major(int N, int K, fp16 *in_b, fp16 *B){
    int i,j,k;
    int m,n;
    for(k=0;k<K;k++)
        for(j=0;j<N;j++){
            m = k*N + j;
            n = j*K + k;
            B[n] = in_b[m];
        }
}

void col2row_major(int N, int K, fp16 *in_b, fp16 *B){
    int i,j,k;
    int m,n;
    for(j=0;j<N;j++)
        for(k=0;k<K;k++){
            m = k*N + j;
            n = j*K + k;
            B[m] = in_b[n];
        }
}


void run(){
    double total,start,end;
    double emu_total;
    cl_mem memobjA = NULL;
    cl_mem memobjB = NULL;
    cl_mem memobjC = NULL;
    const unsigned int caseN=8;
    cl_context context;
    cl_kernel  kernel;
    cl_command_queue command_queue;
/*    float *X =(float*)malloc(6*sizeof(float));
    float *x =(float*)malloc(6*sizeof(float));
    printf("Matrix\n");
    for(int i = 0; i<6 ; i++) X[i]=i/10.;
    for(int i = 0; i<2 ; i++){
        for(int j = 0; j<3 ; j++)
            printf("%f\t",X[i*3+j]);
        printf("\n");
    }
    printf("row-major\n");
    for(int i = 0; i<2*3 ; i++) printf("%f\t",X[i]); printf("\n");
    row2col_major(3,2,X,x);
    printf("col-major\n");
    for(int i = 0; i<2*3 ; i++) printf("%f\t",x[i]); printf("\n");
*/
#ifndef onX86
#ifdef onEMU
    const char *k_name[2]={"gemm_nn9W","gemm_nnfW"};
    cl_kernel kernels[2];
    find_CnKQ(
        "Intel(R) FPGA SDK for OpenCL(TM)",
        "gemm1_emu.aocx",
        2,
        k_name,
        &context, kernels, &command_queue
    );
#else
    const char *k_name[2]={"gemm_nn9W","gemm_nnfW"};
    cl_kernel kernels[2];
    find_CnKQ(
        "Intel(R) FPGA SDK for OpenCL(TM)",
        "gemm1_fpga.aocx",
        2,
        k_name,
        &context, kernels, &command_queue
    );
#endif
#else
    find_CKQ(
        "NVIDIA CUDA",
        "gemm1.cl",
        "gemm_nn4W",
        &context, &kernel, &command_queue
    );
#endif
    cl_int ret1=0,ret2=0,ret3=0;
    float Alpha=1.;
    int ret=0;
    struct caseP{
         int M,N,K;
    }caseP[10];
    caseP[0].M=16;	caseP[0].N=35840;	caseP[0].K=27;
    caseP[1].M=32;	caseP[1].N=8960 ;	caseP[1].K=144;
    caseP[2].M=128;	caseP[2].N=560;		caseP[2].K=288;
    caseP[3].M=512;	caseP[3].N=35;		caseP[3].K=1152;
    caseP[4].M=512;	caseP[4].N=35;		caseP[4].K=4608;
    caseP[5].M=256;	caseP[5].N=35;		caseP[5].K=512;
    caseP[6].M=512;	caseP[6].N=35;		caseP[6].K=2304;
    caseP[7].M=125;	caseP[7].N=35;		caseP[7].K=512;
    /*caseP[0].M=16;	caseP[0].N=32*32;	caseP[0].K=3*3*3;   //  27
    caseP[1].M=32;	caseP[1].N=16*16 ;	caseP[1].K=3*3*16;  // 144
    caseP[2].M=128;	caseP[2].N=8*8;		caseP[2].K=3*3*32;  // 228
    caseP[3].M=512;	caseP[3].N=4*4;		caseP[3].K=3*3*128; //1152
    caseP[4].M=512;	caseP[4].N=2*2;		caseP[4].K=3*3*512; //4608
    caseP[5].M=256;	caseP[5].N=1;		caseP[5].K=1*1*512; // 512
    caseP[6].M=512;	caseP[6].N=35;		caseP[6].K=3*3*256; //2304
    caseP[7].M=125;	caseP[7].N=35;		caseP[7].K=512;*/
    for(int casei=0;casei<caseN;casei++){
        int M=caseP[casei].M;
        int N=caseP[casei].N;
        int K=caseP[casei].K;

        printf("M/N/K = %d\t%d\t%d:\t",M,N,K);

        fp16 *A,*B,*C,*b;
        float *Af, *Bf, *Cf;
        Af=(float*)malloc(sizeof(float)*M*K);
        Bf=(float*)malloc(sizeof(float)*K*N);
        Cf=(float*)malloc(sizeof(float)*M*N);
        for(int x=0;x<M*K;x++)Af[x]=0.1;
        for(int x=0;x<K*N;x++)Bf[x]=1.0;
        for(int x=0;x<M*N;x++)Cf[x]=0.0;
        A=(fp16*)malloc(sizeof(fp16)*M*K);
        B=(fp16*)malloc(sizeof(fp16)*K*N);
        C=(fp16*)malloc(sizeof(fp16)*M*N);
        b=(fp16*)malloc(sizeof(fp16)*K*N);
        for(int x=0;x<M*K;x++)A[x]=f2h(0.1);
        for(int x=0;x<K*N;x++)B[x]=f2h(1.0);
        for(int x=0;x<M*N;x++)C[x]=f2h(0.0);
        if(!(K%27)) kernel = kernels[0];
        else kernel = kernels[1];
        const int nloop=1;
        for(int j=0;j<nloop;j++){

            start = clock_realmsec();
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, Af, K, Bf, N, 0, Cf, N);
            end   = clock_realmsec();
            emu_total = end-start;

            row2col_major(N,K,B,b);
            //memobjA = clCreateBuffer (context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,
            memobjA = clCreateBuffer (context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,
                        M * K * sizeof (cl_half), (void*)A, &ret1);
            checkErr(ret1,"clCreateBuffer0");
            memobjB = clCreateBuffer (context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,
                        K * N * sizeof (cl_half), (void*)b, &ret2);
            checkErr(ret2,"clCreateBuffer1");
            memobjC = clCreateBuffer (context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR,
                        M * N * sizeof (cl_half), (void*)C, &ret3);
            checkErr(ret3,"clCreateBuffer2");

            //clEnqueueWriteBuffer(command_queue,memobjA,CL_TRUE, 0, M*K*sizeof(cl_half),A,0,NULL,NULL);
            //clEnqueueWriteBuffer(command_queue,memobjB,CL_TRUE, 0, K*N*sizeof(cl_half),B,0,NULL,NULL);
            /* Set OpenCL Kernel Parameters */
            ret|= clSetKernelArg (kernel, 0, sizeof (cl_int),  &M); checkErr(ret,"clSetKernelArg-0");
            ret|= clSetKernelArg (kernel, 1, sizeof (cl_int),  &N); checkErr(ret,"clSetKernelArg-1");
            ret|= clSetKernelArg (kernel, 2, sizeof (cl_int),  &K); checkErr(ret,"clSetKernelArg-2");
            ret|= clSetKernelArg (kernel, 3, sizeof (cl_float),&Alpha); checkErr(ret,"clSetKernelArg-3");
            ret|= clSetKernelArg (kernel, 4, sizeof (cl_mem), (void *) &memobjA); checkErr(ret,"clSetKernelArg-4");
            ret|= clSetKernelArg (kernel, 5, sizeof (cl_int),  &K); checkErr(ret,"clSetKernelArg-5");
            ret|= clSetKernelArg (kernel, 6, sizeof (cl_mem), (void *) &memobjB); checkErr(ret,"clSetKernelArg-6");
            ret|= clSetKernelArg (kernel, 7, sizeof (cl_int),  &N); checkErr(ret,"clSetKernelArg-7");
            ret|= clSetKernelArg (kernel, 8, sizeof (cl_mem), (void *) &memobjC); checkErr(ret,"clSetKernelArg-8");
            ret|= clSetKernelArg (kernel, 9, sizeof (cl_int),  &N); checkErr(ret,"clSetKernelArg-9");
            checkErr(ret,"clSetKernelArg");

            /* Execute OpenCL Kernel */
            start = clock_realmsec();
            ret = clEnqueueTask (command_queue, kernel, 0, NULL, NULL);
            checkErr(ret,"clEnqueueTask");
            clFinish(command_queue);
            //clEnqueueReadBuffer(command_queue, memobjC, CL_TRUE, 0, M * N * sizeof(cl_half), (void*)C, 0, NULL, NULL);

            end = clock_realmsec();
            printf(": real time =\t%8.4f(%8.4f) msec\t:",(end-start),emu_total);
            total += (end-start);
            ret = clReleaseMemObject (memobjA);
            ret = clReleaseMemObject (memobjB);
            ret = clReleaseMemObject (memobjC);
            for(int y=0;y<2;y++){
                float Cf = h2f(C[y]);
                printf("%8.4f[%d] ",Cf,y);
            }
            for(int y=0;y<2;y++){
                float Cx = Cf[y];
                printf("%8.4f[%d] ",Cx,y);
            }
            printf("\n");
        }
    }
    printf("real time = %12.6f msec\n:",total);

/* Finalization */
    ret = clFlush (command_queue);
    ret = clFinish (command_queue);
    ret = clReleaseKernel (kernel);
    //  ret = clReleaseProgram (program);
    ret = clReleaseCommandQueue (command_queue);
    ret = clReleaseContext (context);

    //free ((void*)source_str);

}
int main(int argc, char **argv){
    //char *name="NVIDIA CUDA";
    //char *name="Intel(R) FPGA SDK for OpenCL(TM)";
    run();
    exit(0);
}
