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
    double total,start,end;
    cl_mem memobjA = NULL;
    cl_mem memobjB = NULL;
    cl_mem memobjC = NULL;
    const unsigned int caseN=8;
    cl_context context;
    cl_kernel  kernel;
    cl_command_queue command_queue[6];
#ifndef onX86
#ifdef onEMU
    const char *k_name[3]={"mem_read","mem_write","conv"};
    cl_kernel kernels[3];
    find_CnKQ(
        "Intel(R) FPGA SDK for OpenCL(TM)",
        "memcp_emu.aocx",
        2,
        k_name,
        &context, kernels, command_queue
    );
#else
    const char *k_name[3]={"mem_read","mem_write","conv"};
    cl_kernel kernels[3];
    find_CnKQ(
        "Intel(R) FPGA SDK for OpenCL(TM)",
        //"sr1f_fpga.aocx",
        //"sr36_fpga.aocx",
        //"sr1ftdef_fpga.aocx",
        //"memcp_fpga.aocx",
        "memcpdef32_fpga.aocx",
        2,
        k_name,
        &context, kernels, command_queue
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
    int ret=0;
    int i,j,k;
    for(int casei=0;casei<caseN;casei++){
        int M=224*160*3;
        int N=224*160*16;
        int K=224*160*3;

        printf("M/N/K = %d\t%d\t%d:\t",M,N,K);

        float *A,*B,*C;
        A=(float*)malloc(M*sizeof(float));
        B=(float*)malloc(N*sizeof(float));
        C=(float*)malloc(K*sizeof(float));
        for(int x=0;x<M;x++)A[x]=0.;
        for(int x=0;x<N;x++)B[x]=x+1.;
        for(int x=0;x<K;x++)C[x]=x+2.;
        const int nloop=1;
        for(int j=0;j<nloop;j++){

            memobjA = clCreateBuffer (context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,
                        M * sizeof (cl_float), (void*)A, &ret1); checkErr(ret1,"clCreateBuffer0");

            memobjB = clCreateBuffer (context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR,
                        N * sizeof (cl_float), (void*)B, &ret2); checkErr(ret2,"clCreateBuffer1");

            memobjC = clCreateBuffer (context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR,
                        K * sizeof (cl_float), (void*)C, &ret3); checkErr(ret3,"clCreateBuffer2");

            /* Set OpenCL Kernel Parameters */
            ret|= clSetKernelArg (kernels[0], 0, sizeof (cl_int), (void *) &M);       checkErr(ret,"clSetKernelArg-0");
            ret|= clSetKernelArg (kernels[0], 1, sizeof (cl_mem), (void *) &memobjA); checkErr(ret,"clSetKernelArg-1");
            ret|= clSetKernelArg (kernels[1], 0, sizeof (cl_int), (void *) &M);       checkErr(ret,"clSetKernelArg-2");
            ret|= clSetKernelArg (kernels[1], 1, sizeof (cl_mem), (void *) &memobjB); checkErr(ret,"clSetKernelArg-3");
            //ret|= clSetKernelArg (kernel, 9, sizeof (cl_int),  &N); checkErr(ret,"clSetKernelArg-9");

            /* Execute OpenCL Kernel */
            start = clock_realmsec();
            size_t wgrp[]={M,0,0};
            size_t lgrp[]={M,0,0};
            ret = clEnqueueTask (command_queue[0], kernels[0], 0, NULL, NULL);  checkErr(ret,"clEnqueueTask-0");
            ret = clEnqueueTask (command_queue[1], kernels[1], 0, NULL, NULL);  checkErr(ret,"clEnqueueTask-1");
            //ret = clEnqueueNDRangeKernel(command_queue[0], kernels[0], 1, 0, wgrp,lgrp, 0, NULL, NULL);
            clFinish(command_queue[1]);
            //clEnqueueReadBuffer(command_queue[2], memobjC, CL_TRUE, 0, K * sizeof(cl_float), (void*)C, 0, NULL, NULL);

            end = clock_realmsec();
            printf(":\treal time = %8.3f msec\t:",(end-start));
            for(i=2*224+3-1,j=0;i<M;i++)
                if(B[i] != A[i]){
                    printf("diff->B[%d] A[%d] = %5.2f %5.2f ",i,i,B[i],A[i]);
                    break;
                }else j++;
            total += (end-start);
            ret = clReleaseMemObject (memobjA);
            ret = clReleaseMemObject (memobjB);
            ret = clReleaseMemObject (memobjC);
            printf("%d match %6.3fMBPS\n",j,1.0*M*sizeof(float)/(1000.*(end-start)));
        }
        free(A);
        free(B);
        free(C);
    }
    printf("real time = %12.6f msec\n:",total);

/* Finalization */
    //ret = clFinish (command_queue[1]);
    //ret = clReleaseKernel (kernel);
    //ret = clReleaseProgram (program);
    //ret = clReleaseCommandQueue (command_queue[0]);
    //ret = clReleaseCommandQueue (command_queue[1]);
    //ret = clReleaseCommandQueue (command_queue[2]);
    ret = clReleaseContext (context);
}
int main(int argc, char **argv){
    //char *name="NVIDIA CUDA";
    //char *name="Intel(R) FPGA SDK for OpenCL(TM)";
    run();
    exit(0);
}
