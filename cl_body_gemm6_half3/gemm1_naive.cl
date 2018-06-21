#pragma OPENCL EXTENSION cl_khr_fp16 : enable
kernel void gemm_nn4W (const int M, const int N, const int K, const float ALPHA,
		 global half *restrict A, const int lda,
		 global half *restrict B, const int ldb,
		 global half *restrict C, const int ldc
		)
{
  int i, j, k, m;
  half A_PART;
  for (i = 0; i < M; ++i) {
	for (k = 0; k < K; ++k) {
	  A_PART = A[i * lda + k];
	  for (j = 0; j < N; ++j) {
		C[i * ldc + j]+= A_PART * B[k * ldb + j];
	  }
	}
  }
}
