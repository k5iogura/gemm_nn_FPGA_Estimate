#define __USE_MINGW_ANSI_STDIO 1
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x500000)

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

void gemm_nn4W (int M, int N, int K, float ALPHA,
		 float * A, int lda,
		 float * B, int ldb,
		 float * C, int ldc
		)
{
  int i, j, k, m;
  float A_PART;
  for (i = 0; i < M; ++i) {
	for (k = 0; k < K; ++k) {
	  A_PART = A[i * lda + k];
	  for (j = 0; j < N; ++j) {
		C[i * ldc + j]+= A_PART * B[k * ldb + j];
	  }
	}
  }
}

int
main ()
{
  cl_device_id device_id = NULL;
  cl_context context = NULL;
  cl_command_queue command_queue = NULL;
  cl_mem memobjA = NULL;
  cl_mem memobjB = NULL;
  cl_mem memobjC = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_platform_id platform_id = NULL;
  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_int ret;
  cl_int ret1,ret2,ret3;
  float Alpha=1.0;

  double total=0,start,end;

  FILE *fp;
#ifdef onX86
  char fileName[] = "./gemm1.cl";
#else
  char fileName[] = "./gemm1.aocx";
#endif
  const unsigned char *source_str;
  size_t source_size;

/* Load the source code containing the kernel*/
  fp = fopen (fileName, "r");
  if (!fp) {
	fprintf (stderr, "Failed to load kernel.\n");
	exit (1);
  }else printf("fileName=%s\n",fileName);
  source_str = (const unsigned char *) malloc (MAX_SOURCE_SIZE);
  source_size = fread ((void*)source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose (fp);

/* Get Platform and Device Info */
  ret = clGetPlatformIDs (1, &platform_id, &ret_num_platforms);
  if(ret != CL_SUCCESS){
	fprintf(stderr,"Faild clGetPlatform %d\n",ret);
	exit(ret);
  }else{fprintf(stderr,"CL_SUCCESS 0-0\n");}
  ret =
	clGetDeviceIDs (platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id,
					&ret_num_devices);
  if(ret != CL_SUCCESS){
	fprintf(stderr,"Faild clGetDeviceIDs %d\n",ret);
	exit(ret);
  }else{fprintf(stderr,"CL_SUCCESS 0-1\n");}
  cl_ulong local_mem;
  clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong),&local_mem,NULL);
  printf("LOCAL_MEM_SIZE=%lu\n",local_mem);

/* Create OpenCL context */
  context = clCreateContext (NULL, 1, &device_id, NULL, NULL, &ret);
  if(ret != CL_SUCCESS){
	fprintf(stderr,"Faild clCreateContext %d\n",ret);
	exit(ret);
  }else{fprintf(stderr,"CL_SUCCESS 0-2\n");}

/* Create Command Queue */
  command_queue = clCreateCommandQueue (context, device_id, 0, &ret);
  if(ret != CL_SUCCESS){
	fprintf(stderr,"Faild clCreateCommandQueue %d\n",ret);
	exit(ret);
  }else{fprintf(stderr,"CL_SUCCESS 0-3\n");}

/* Create Kernel Program from the source */
#ifdef onX86
  printf("load from source onX86\n");
  program =
	clCreateProgramWithSource (context, 1, (const char **) &source_str,
							   (const size_t *) &source_size, &ret);
#else
  printf("load from binary onARM\n");
  program =
	clCreateProgramWithBinary (context, 1, &device_id, &source_size,
		(const unsigned char **) &source_str, &ret1, &ret);
#endif
  if(ret != CL_SUCCESS){
	fprintf(stderr,"Faild clCreateProgramWithXXXX %d\n",ret);
	exit(ret);
  }else{fprintf(stderr,"CL_SUCCESS 1\n");}

/* Build Kernel Program */
  ret = clBuildProgram (program, 1, &device_id, NULL, NULL, NULL);
  if(ret != CL_SUCCESS){
	fprintf(stderr,"Faild clBuildProgram %d\n",ret);
	exit(ret);
  }else{fprintf(stderr,"CL_SUCCESS 2\n");}

  cl_kernel kernels[10];
  cl_uint n_kernels=1;
  ret = clCreateKernelsInProgram(program,1,kernels,&n_kernels);
  printf("In Program kernels = %d\n",n_kernels);
  if(ret != CL_SUCCESS){
	fprintf(stderr,"Faild clCreateKernelsInProgram %d\n",ret);
//	exit(ret);
  }else{

	  char name[64];
	  size_t info_size;
	  for(int i=0;i<n_kernels;i++){
		  ret = clGetKernelInfo(kernels[i],CL_KERNEL_FUNCTION_NAME,32,name,&info_size);
		  printf("In Program kernel[%d] name = %s\n",i,name);
		  clReleaseKernel(kernels[i]);
	  }
  }

/* Create OpenCL Kernel */
  //char kernel_name[128]="gemm_nn";
  //char kernel_name[128]="gemm_nnA1";
  //char kernel_name[128]="gemm_nnWB";
  char kernel_name[128]="gemm_nn4W";
  kernel = clCreateKernel (program, kernel_name, &ret);
  if(ret != CL_SUCCESS){
	fprintf(stderr,"Faild clCreateKernel %d\n",ret);
	exit(ret);
  }else{fprintf(stderr,"CL_SUCCESS 4 %s\n",kernel_name);}

  const unsigned int caseN=4;
  struct caseP{
     int M,N,K;
  }caseP[10];
  //int M=16,N=1024,K=144;	// max in cifar10 dataset
  //int M=48,N=64,K=48;		// min in cifar10 dataset
  //int M=32,N=256,K=288;	// 2nd in cifar10 dataset
  //int M=48,N=64,K=432;	// 3rd in cifar10 dataset
  //int M=16,N=1024,K=16;		// 1x1 in max
  //int M=32,N=12544,K=144;
  caseP[0].M=16;	caseP[0].N=35840;	caseP[0].K=27;
  caseP[1].M=32;	caseP[1].N=8960 ;	caseP[1].K=144;
  caseP[2].M=128;	caseP[2].N=560;		caseP[2].K=288;
  caseP[3].M=512;	caseP[3].N=35;		caseP[3].K=1152;
  caseP[4].M=512;	caseP[4].N=35;		caseP[4].K=4608;
  caseP[5].M=256;	caseP[5].N=35;		caseP[5].K=512;
  caseP[6].M=512;	caseP[6].N=35;		caseP[6].K=2304;
  caseP[7].M=125;	caseP[7].N=35;		caseP[7].K=512;
  caseP[8].M=5;	caseP[8].N=5;	    caseP[8].K=5;
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
	  memobjA = clCreateBuffer (context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,
						M * K * sizeof (float), A, &ret1);
	  memobjB = clCreateBuffer (context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,
						K * N * sizeof (float), B, &ret2);
	  memobjC = clCreateBuffer (context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR,
						M * N * sizeof (float), C, &ret3);
//	  if(ret1 != CL_SUCCESS || ret2 != CL_SUCCESS || ret3 != CL_SUCCESS){
//		fprintf(stderr,"Faild clCreateBuffer %d %d %d\n",ret1,ret2,ret3);
//		exit(ret);
//	  }else{fprintf(stderr,"CL_SUCCESS 4\n");}

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
	  //ret|= clSetKernelArg (kernel,10, sizeof(cl_float)*K*N,  NULL); // about 1450msec at local memory
//	  about 1437msec at private memory
	  if(ret != CL_SUCCESS){
		fprintf(stderr,"Faild clSetKernelArg %d\n",ret);
		exit(ret);
	  }else{fprintf(stderr,"CL_SUCCESS 5\n");}

	/* Execute OpenCL Kernel */
	  //ret = clEnqueueTask (command_queue, kernel, 0, NULL, NULL);
	  start = clock_realmsec();
      gemm_nn4W (M, N, K, Alpha,
               A, K,
               B, N,
               C, N
      );
      end = clock_realmsec();
//	  if(ret != CL_SUCCESS){
//		fprintf(stderr,"Faild clEnqueueTask %d\n",ret);
//		exit(ret);
//	  }else{fprintf(stderr,"CL_SUCCESS 6\n");}

	/* Copy results from the memory buffer */
	//  ret = clEnqueueReadBuffer (command_queue, memobj, CL_TRUE, 0,
	//					MEM_SIZE * sizeof (char), string, 0, NULL, NULL);
	  //clFinish(command_queue);
	  //if(ret == CL_SUCCESS){
		  for(int y=0;y<1;y++)
			printf("%f[%d]\n",C[y],y);

		  //ret = clReleaseMemObject (memobjA);
		  //ret = clReleaseMemObject (memobjB);
		  //ret = clReleaseMemObject (memobjC);
		  printf("real time = %.3fmsec\n",(end-start));
      total+=(end-start);
	  //}else{fprintf(stderr,"clEnqueueTask Error %d\n",ret);break;}
	}
	}
  printf("total time = %.3fmsec\n",(total));

/* Finalization */
  ret = clFlush (command_queue);
  ret = clFinish (command_queue);
  ret = clReleaseKernel (kernel);
  ret = clReleaseProgram (program);
  ret = clReleaseCommandQueue (command_queue);
  ret = clReleaseContext (context);

  free ((void*)source_str);

  return 0;
}
