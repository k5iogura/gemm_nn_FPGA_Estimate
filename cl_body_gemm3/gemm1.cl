float sum4(float4 a){
    return
    a.s0 +
    a.s1 +
    a.s2 +
    a.s3 ;
}
float dot4(float4 a, float4 b){
    return
    a.s0*b.s0 +
    a.s1*b.s1 +
    a.s2*b.s2 +
    a.s3*b.s3 ;
}
#define WRD (4)
kernel void gemm_nn4W (int M, int N, int K, float ALPHA,
		 global float *restrict A, int lda,
		 global float *restrict B, int ldb,
		 global float *restrict C, int ldc
		)
{
  int i, j, k;
  float A_PART;
  float4 Ax, Bx, Cx;
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
        Ax = vload4((i*wlda+k), A);
        Bx = vload4((j*wlda+k), B);
        //Cn+= dot4(Bx , Ax);
        Cx = Bx * Ax;
        Cn+= sum4(Cx);
	  }
      C[ i*ldc + j ] = Cn;
	}
  }
  //printf("%f\n",Cx);
}
