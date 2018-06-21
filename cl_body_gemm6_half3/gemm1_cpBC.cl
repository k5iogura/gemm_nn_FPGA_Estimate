#define WRD (35)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
kernel void gemm_nn4W (const int M, const int N, const int K, const float ALPHA,
		 global half *restrict A, const int lda,
		 global half *restrict B, const int ldb,
		 global half *restrict C, const int ldc
		)
{
  int i, j, k, m;
  int wldb = ldb/WRD;
  int wldc = ldc/WRD;
  half A_PART;
  for (i = 0; i < M; ++i) {
	for (k = 0; k < K; ++k) {
	  A_PART = A[i * lda + k];
	  for (j = 0; j < N/WRD; ++j) {
        float16 Bx1 = vload_half16(( k*wldb + j +  0), B);
        float16 Bx2 = vload_half16(( k*wldb + j + 16), B);
        float3  Bx3 = vload_half3( ( k*wldb + j + 32), B);
        vstore_half16( Bx1, ( i*wldc + j +  0), C);
        vstore_half16( Bx2, ( i*wldc + j + 16), C);
        vstore_half3(  Bx3, ( i*wldc + j + 32), C);
	  }
	}
  }
}
