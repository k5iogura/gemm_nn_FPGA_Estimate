kernel void gemm_nn4W (const int M, const int N, const int K, const float ALPHA,
		 global float *restrict A, const int lda,
		 global float *restrict B, const int ldb,
		 global float *restrict C, const int ldc,
         const int i, const int k, const int j
		)
{
  size_t J = get_global_id(0);
  C[i * ldc + J]+= A_PART * B[k * ldb + J];
}
