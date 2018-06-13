kernel void gemm_nn4W (const int M, const int N, const int K, const float ALPHA,
		 global float *restrict A, const int lda,
		 global float *restrict B, const int ldb,
		 global float *restrict C, const int ldc
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
