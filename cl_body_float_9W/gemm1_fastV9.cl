float sum3(float3 a){
    return
    a.s0 +
    a.s1 +
    a.s2 ;
}
float sum8(float8 a){
    return
    a.s0 +
    a.s1 +
    a.s2 +
    a.s3 +
    a.s4 +
    a.s5 +
    a.s6 +
    a.s7 ;
}
float sum16(float16 a){
    return
    a.s0 +
    a.s1 +
    a.s2 +
    a.s3 +
    a.s4 +
    a.s5 +
    a.s6 +
    a.s7 +
    a.s8 +
    a.s9 +
    a.sa +
    a.sb +
    a.sc +
    a.sd +
    a.se +
    a.sf ;
}
#define WRD (9)
kernel void gemm_nn4W (int M, int N, int K, float ALPHA,
		 global float *restrict A, int lda,
		 global float *restrict B, int ldb,
		 global float *restrict C, int ldc
		)
{
  int i, j, k;
  float A_PART;
  //float8 Ax, Bx, Cx;
  //float  A1, B1, C1;
  //float Cn;
  int wM = M/WRD;
  int wN = N/WRD;
  int wK = K/WRD;
  int wlda = lda/WRD;
  int wldb = ldb/WRD;
  int wldc = ldc/WRD;
  for (i = 0; i < M; ++i) {
    for (j = 0; j < N; ++j) {
      float Cn = C[ i*ldc + j ];
	  for (k = 0; k < wK; ++k) {
        float8 Ax = vload8(( i*wlda + k), A);
        float  A1 = A[ i*wlda + k + 8 ];
        float8 Bx = vload8(( j*wlda + k), B);
        float  B1 = B[ j*wlda + k + 8 ];
        float8 Cx = Bx * Ax;
        float  C1 = B1 * A1;
        Cn+= sum8(Cx)+C1;
	  }
      C[ i*ldc + j ] = Cn;
	}
  }
}
