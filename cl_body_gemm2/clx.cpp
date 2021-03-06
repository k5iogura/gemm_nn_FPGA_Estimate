#define __USE_MINGW_ANSI_STDIO 1
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

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

void run(){
    int i,j,k;
    double start,end;
    cl_mem memobjA = NULL;
    cl_mem memobjB = NULL;
    cl_mem memobjC = NULL;
    const unsigned int caseN=9;
    cl_context context;
    cl_kernel  kernel;
    cl_command_queue command_queue;
#ifndef onX86
#ifdef onEMU
    find_CKQ(
        "Intel(R) FPGA SDK for OpenCL(TM)",
        "gemm1_emu.aocx",
        "gemm_nn4W",
        &context, &kernel, &command_queue
    );
#else
    find_CKQ(
        "Intel(R) FPGA SDK for OpenCL(TM)",
        "gemm1_fpga.aocx",
        "gemm_nn4W",
        &context, &kernel, &command_queue
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
    //int M=16,N=1024,K=144;	// max in cifar10 dataset
    //int M=48,N=64,K=48;		// min in cifar10 dataset
    //int M=32,N=256,K=288;	// 2nd in cifar10 dataset
    //int M=48,N=64,K=432;	// 3rd in cifar10 dataset
    //int M=16,N=1024,K=16;		// 1x1 in max
    caseP[0].M=16;	caseP[0].N=35840;	caseP[0].K=27;
    caseP[1].M=32;	caseP[1].N=8960 ;	caseP[1].K=144;
    caseP[2].M=128;	caseP[2].N=560;		caseP[2].K=288;
    caseP[3].M=512;	caseP[3].N=35;		caseP[3].K=1152;
    caseP[4].M=512;	caseP[4].N=35;		caseP[4].K=4608;
    caseP[5].M=256;	caseP[5].N=35;		caseP[5].K=512;
    caseP[6].M=512;	caseP[6].N=35;		caseP[6].K=2304;
    caseP[7].M=125;	caseP[7].N=35;		caseP[7].K=512;
    caseP[8].M=35;	caseP[8].N=35;	    caseP[8].K=35;
    //int M=32,N=12544,K=144;
    //int M=16,N=3136,K=32;
    //int M=512,N=196,K=576;
    //int M=1024,N=169,K=9216;
    //  FILE *ff=fopen("gemmX.txt","r");
    //  if(!ff) printf("error fopen\n"),exit(-1);
    //  fscanf(ff,"%d\t%d\t%d",&M,&N,&K);
    for(int casei=0;casei<caseN;casei++){
        int M=caseP[casei].M;
        int N=caseP[casei].N;
        int K=caseP[casei].K;

        printf("M/N/K = %d\t%d\t%d:\t",M,N,K);

        float *A,*B,*C;
        A=(float*)malloc(sizeof(float)*M*K);
        B=(float*)malloc(sizeof(float)*K*N);
        C=(float*)malloc(sizeof(float)*M*N);
        for(int x=0;x<M*K;x++)A[x]=2.0;
        for(int x=0;x<K*N;x++)B[x]=3.0;
        for(int x=0;x<M*N;x++)C[x]=0.0;
        const int nloop=1;
        for(int j=0;j<nloop;j++){

            memobjA = clCreateBuffer (context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,
                        M * K * sizeof (float), A, &ret1);
            checkErr(ret1,"clCreateBuffer0");
            memobjB = clCreateBuffer (context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,
                        K * N * sizeof (float), B, &ret2);
            checkErr(ret2,"clCreateBuffer1");
            memobjC = clCreateBuffer (context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR,
                        M * N * sizeof (float), C, &ret3);
            checkErr(ret3,"clCreateBuffer2");

            //clEnqueueWriteBuffer(command_queue,memobjA,CL_TRUE, 0, M*K*4,A,0,NULL,NULL);
            //clEnqueueWriteBuffer(command_queue,memobjB,CL_TRUE, 0, K*N*4,B,0,NULL,NULL);
            /* Set OpenCL Kernel Parameters */
            ret|= clSetKernelArg (kernel, 0, sizeof (cl_int),  &M);
            ret|= clSetKernelArg (kernel, 1, sizeof (cl_int),  &N);
            ret|= clSetKernelArg (kernel, 2, sizeof (cl_int),  &K);
            ret|= clSetKernelArg (kernel, 3, sizeof (cl_float),&Alpha);
            ret|= clSetKernelArg (kernel, 4, sizeof (cl_mem), (void *) &memobjA);
            ret|= clSetKernelArg (kernel, 5, sizeof (cl_int),  &K);
            ret|= clSetKernelArg (kernel, 6, sizeof (cl_mem), (void *) &memobjB);
            ret|= clSetKernelArg (kernel, 7, sizeof (cl_int),  &N);
            ret|= clSetKernelArg (kernel, 8, sizeof (cl_mem), (void *) &memobjC);
            ret|= clSetKernelArg (kernel, 9, sizeof (cl_int),  &N);
            checkErr(ret,"clSetKernelArg");

            /* Execute OpenCL Kernel */
            start = clock_realmsec();
            size_t wgrp[]={N,0,0};
            size_t lgrp[]={35,0,0};
            //gemm
            //float A_PART;
            for (i = 0; i < M; ++i) {
                for (k = 0; k < K; ++k) {
                    ret|= clSetKernelArg (kernel,10, sizeof (cl_int),  &i);
                    ret|= clSetKernelArg (kernel,11, sizeof (cl_int),  &k);
                    ret|= clSetKernelArg (kernel,12, sizeof (cl_int),  &j);
                    //A_PART = A[i * lda + k];
                    //for (j = 0; j < N; ++j) {
                        //C[i * ldc + j]+= A_PART * B[k * ldb + j];
                        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, 0, wgrp,lgrp, 0, NULL, NULL);
                    //}
                        checkErr(ret,"clEnqueueNDRangeKernel");
                }
            }
            //ret = clEnqueueTask (command_queue, kernel, 0, NULL, NULL);
            clFinish(command_queue);
            clEnqueueReadBuffer(command_queue, memobjC, CL_TRUE, 0, M * N * sizeof(float), (void*)C, 0, NULL, NULL);

            end = clock_realmsec();
            printf(":\treal time = %12.6f msec\t:",(end-start));
            ret = clReleaseMemObject (memobjA);
            ret = clReleaseMemObject (memobjB);
            ret = clReleaseMemObject (memobjC);
            for(int y=0;y<3;y++)
                printf("%f[%d]\t",C[y],y);
            printf("\n");
        }
    }

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
