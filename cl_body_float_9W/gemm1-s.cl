#define WRD (8)
float dot8(float8 a, float8 b){
    return a.s0*b.s0 +
    a.s1*b.s1 +
    a.s2*b.s2 +
    a.s3*b.s3 +
    a.s4*b.s4 +
    a.s5*b.s5 +
    a.s6*b.s6 +
    a.s7*b.s7 ;
}
kernel void gemm_nn4W (const int M, const int N, const int K, const float ALPHA,
		 global float *restrict A, const int lda,
		 global float *restrict B, const int ldb,
		 global float *restrict C, const int ldc
		)
{
  int i, j, k;
  float A_PART;
  float8 Ax, Bx, Cx;
  int wM = M/WRD;
  int wN = N/WRD;
  int wK = K/WRD;
  int wlda = lda/WRD;
  int wldb = ldb/WRD;
  int wldc = ldc/WRD;
  int countn = 0;
  for (i = 0; i < wM; ++i) {
    for (j = 0; j < wN; j++) {
      float C1=0;
      if(!countn) Cx = vload8((i*wldc+j), C);
	  for (k = 0; k < wK; ++k) {
        Ax = vload8((i*wlda+k), A);
        Bx = vload8((k*wldb+j), B);
        C1+= dot8(Bx , Ax);
	  }
      switch (countn){
        case 0:
          Cx.s0 = C1;break;
        case 1:
          Cx.s1 = C1;break;
        case 2:
          Cx.s2 = C1;break;
        case 3:
          Cx.s3 = C1;break;
        case 4:
          Cx.s4 = C1;break;
        case 5:
          Cx.s5 = C1;break;
        case 6:
          Cx.s6 = C1;break;
        case 7:
          Cx.s7 = C1;break;
        default:
          //vstore8(Cx,(i*wldc+j),C);
          break;
      }
      if(countn==7)
        countn=0;
      else
        countn++;
	}
  }
}
