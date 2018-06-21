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
#pragma ii 16
	  for (j = 0; j < N/WRD; ++j) {
        float16 Bx1 = vload_half16(( k*wldb + j +  0), B);
        float16 Bx2 = vload_half16(( k*wldb + j + 16), B);
        float3  Bx3 = vload_half3( ( k*wldb + j + 32), B);
        float16 Cx1 = vload_half16(( i*wldc + j +  0), C);
        float16 Cx2 = vload_half16(( i*wldc + j + 16), C);
        float3  Cx3 = vload_half3( ( i*wldc + j + 32), C);
        Cx1+= A_PART * Bx1;
        Cx2+= A_PART * Bx2;
        Cx3+= A_PART * Bx3;
        vstore_half16( Cx1, ( i*wldc + j +  0), C);
        vstore_half16( Cx2, ( i*wldc + j + 16), C);
        vstore_half3(  Cx3, ( i*wldc + j + 32), C);
	  }
	}
  }
}
