kernel void gemm_nn4W (const int M, const int N, const int K, const float ALPHA,
		 global float *restrict A, const int lda,
		 global float *restrict B, const int ldb,
		 global float *restrict C, const int ldc
		)
{
  int i, j, k, m;
  float A_PART;
  float Bxx[35840];
  bool  update=true;
  for (k = 0; k < K; ++k) {
    for (i = 0; i < M; ++i) {
	  A_PART = A[i * lda + k];
      for (j = 0; j < N; ++j) Bxx[j] = (update)? B[ k*ldb + j ]:Bxx[j];
      for (j = 0; j < N; ++j) {
		C[i * ldc + j]+= A_PART * Bxx[j];
	  }
      update=false;
	}
    update=true;
  }
}
