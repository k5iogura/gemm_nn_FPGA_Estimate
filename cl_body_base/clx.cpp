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
//	timeval tv;
//	timezone tz;
//	gettimeofday(&tv,&tz);
//	tv.tv_sec=tv.tv_usec=0;
//	settimeofday(&tv,&tz);
}
double clock_realmsec(){
	timespec ts;
	clock_gettime(CLOCK_MONOTONIC,&ts);
//	clock_gettime(CLOCK_REALTIME,&ts);
	return (ts.tv_sec+ts.tv_nsec*1e-9)*1000.;
//	timeval tv;
//	timezone tz;
//	gettimeofday(&tv,&tz);
//	return ((float)tv.tv_sec);
}

void run(){
  double start,end;
  cl_mem memobjA = NULL;
  cl_mem memobjB = NULL;
  cl_mem memobjC = NULL;
  const unsigned int caseN=4;
  cl_context context;
  cl_kernel  kernel;
  cl_command_queue command_queue;
  find_CKQ(
    "Intel(R) FPGA SDK for OpenCL(TM)",
    "gemm1.aocx",
    "gemm_nn4W",
    &context, &kernel, &command_queue
  );
  cl_int ret1,ret2,ret3;
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
  caseP[0].M=16;	caseP[0].N=1024;	caseP[0].K=144;
  caseP[1].M=48;	caseP[1].N=121 ;	caseP[1].K=48;
  caseP[2].M=32;	caseP[2].N=121;		caseP[2].K=32;
  caseP[3].M=3 ;	caseP[3].N=16;		caseP[3].K=3;
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

  printf("M/N/K = %d\t%d\t%d\n",M,N,K);

  float *A,*B,*C;
  A=(float*)malloc(sizeof(float)*M*K);
  B=(float*)malloc(sizeof(float)*K*N);
  C=(float*)malloc(sizeof(float)*M*N);
  for(int x=0;x<M*K;x++)A[x]=1.0;
  for(int x=0;x<K*N;x++)B[x]=2.0;
  for(int x=0;x<M*N;x++)C[x]=0.0;
	const int nloop=1;
	for(int j=0;j<nloop;j++){
	  start = clock_realmsec();
	  memobjA = clCreateBuffer (context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,
						M * K * sizeof (float), A, &ret1);
	  memobjB = clCreateBuffer (context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,
						K * N * sizeof (float), B, &ret2);
	  memobjC = clCreateBuffer (context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR,
						M * N * sizeof (float), C, &ret3);

	/* Set OpenCL Kernel Parameters */
	  ret|= clSetKernelArg (kernel, 0, sizeof (cl_int),  &M);
	  ret|= clSetKernelArg (kernel, 1, sizeof (cl_int),  &N);
	  ret|= clSetKernelArg (kernel, 2, sizeof (cl_int),  &K);
	  ret|= clSetKernelArg (kernel, 3, sizeof (cl_float),   &Alpha);
	  ret|= clSetKernelArg (kernel, 4, sizeof (cl_mem), (void *) &memobjA);
	  ret|= clSetKernelArg (kernel, 5, sizeof (cl_int),  &K);
	  ret|= clSetKernelArg (kernel, 6, sizeof (cl_mem), (void *) &memobjB);
	  ret|= clSetKernelArg (kernel, 7, sizeof (cl_int),  &N);
	  ret|= clSetKernelArg (kernel, 8, sizeof (cl_mem), (void *) &memobjC);
	  ret|= clSetKernelArg (kernel, 9, sizeof (cl_int),  &N);
	  if(ret != CL_SUCCESS){
		fprintf(stderr,"Faild clSetKernelArg %d\n",ret);
		exit(ret);
	  }else{fprintf(stderr,"CL_SUCCESS 5\n");}

	/* Execute OpenCL Kernel */
	  ret = clEnqueueTask (command_queue, kernel, 0, NULL, NULL);
	  clFinish(command_queue);
	  if(ret == CL_SUCCESS){
		  for(int y=0;y<1;y++)
			printf("%f[%d]\n",C[y],y);

		  ret = clReleaseMemObject (memobjA);
		  ret = clReleaseMemObject (memobjB);
		  ret = clReleaseMemObject (memobjC);
		  end = clock_realmsec();
		  printf("real time = %.3fmsec\n",(end-start));
	  }else{fprintf(stderr,"clEnqueueTask Error %d\n",ret);break;}
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
    char *name="Intel(R) FPGA SDK for OpenCL(TM)";
    run();
    exit(0);
}
