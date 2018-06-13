kernel void gemm_nn4W (const int M, const int N, const int K, const float ALPHA,
		 global float *restrict A, const int lda,
		 global float *restrict B, const int ldb,
		 global float *restrict C, const int ldc, const int i, const int k, const int j
		)
{
  int J;
  local float a_A;
  J = get_global_id(0);
  if(!get_local_id(0))
      a_A = A[i * lda + k];
  barrier(CLK_LOCAL_MEM_FENCE);
  C[i * ldc + J]+= a_A * B[k * ldb + J];
}
