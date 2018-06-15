#define WRD (16)
kernel void gemm_nn4W (int M, int N, int K, float ALPHA,
		 global float *restrict A, int lda,
		 global float *restrict B, int ldb,
		 global float *restrict C, int ldc
		)
{
  int i, j, k;
  float A_PART;
  float16 Ax, Bx,Cx;
  M = M/WRD;
  N = N/WRD;
  K = K/WRD;
  lda = lda/WRD;
  ldb = ldb/WRD;
  ldc = ldc/WRD;
  for (i = 0; i < M; ++i) {
    for (j = 0; j < N; j++) {
      Cx = vload16((i*ldc+j), B);
	  for (k = 0; k < K; ++k) {
        Ax = vload16((i*lda+k), A);
        Bx = vload16((k*ldb+j), B);
      //  Cx = Bx;
      //  Cx+= Bx * Ax;
	  }
      vstore16(Cx,(i*ldc+j),C);
	}
  }
}
