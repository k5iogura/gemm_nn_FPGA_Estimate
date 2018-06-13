kernel void gemm_nn4W (const int M, const int N, const int K, const float ALPHA,
		 global float *restrict A, const int lda,
		 global float *restrict B, const int ldb,
		 global float *restrict C, const int ldc, const int i, const int k, const int j
		)
{
  int J;
  J = get_global_id(0);
  C[i * ldc + J]+= ALPHA * B[k * ldb + J];
}
