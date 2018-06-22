#define WRD (8)
kernel void gemm_nn4W (int M, int N, int K, float ALPHA,
		 global float *restrict A, int lda,
		 global float *restrict B, int ldb,
		 global float *restrict C, int ldc
		)
{
  int i, j, k;
  float A_PART;
  float8 Ax, Bx,Cx;
  M = M/WRD;
  N = N/WRD;
  K = K/WRD;
  lda = lda/WRD;
  ldb = ldb/WRD;
  ldc = ldc/WRD;
  for (i = 0; i < M; ++i) {
    for (j = 0; j < N; j++) {
	  for (k = 0; k < K; ++k) {
        Ax = vload8((i*lda+k), A);
        Bx = vload8((k*ldb+j), B);
        Cx = vload8((i*ldc+j), B);
        Cx+= Bx * Ax;
	  }
      vstore8(Cx,(i*ldc+j),C);
	}
  }
}
