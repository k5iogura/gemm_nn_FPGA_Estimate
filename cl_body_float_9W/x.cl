//__attribute__((num_compute_units(2)))
#define WRD (16)
kernel void gemm_nn4W (const int M, const int N, const int K, const float ALPHA,
		 global float *restrict A, const int lda,
		 global float *restrict B, const int ldb,
		 global float *restrict C, const int ldc, const int i, const int dk, const int dj
		)
{
  int j, k;
  float A_PART;
  for (k = 0; k < K; ++k) {
    A_PART = A[i * lda + k];
    for (j = 0; j < N; j+=WRD) {
      float16 Bx = vload16((k*ldb+j)/WRD, B);
      float16 Cx = vload16((i*ldc+j)/WRD, C);
      Cx+= Bx * A_PART;
      vstore16(Cx, (i*ldc+j)/WRD, C);
    }
  }
}
