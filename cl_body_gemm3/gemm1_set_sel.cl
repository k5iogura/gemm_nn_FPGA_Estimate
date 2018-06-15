#define WRD (8)
float dot8(float8 a, float8 b){
    return
    a.s0*b.s0 +
    a.s1*b.s1 +
    a.s2*b.s2 +
    a.s3*b.s3 +
    a.s4*b.s4 +
    a.s5*b.s5 +
    a.s6*b.s6 +
    a.s7*b.s7 ;
}
float sel_float8(float8 f, unsigned char e){
    switch (e){
      case 0: return f.s0;
      case 1: return f.s1;
      case 2: return f.s2;
      case 3: return f.s3;
      case 4: return f.s4;
      case 5: return f.s5;
      case 6: return f.s6;
      case 7: return f.s7;
      default: return 0;
    }
}
float8 set_float8(float8 f, unsigned char e, float v){
    float8 dummy=0;
    switch (e){
      case 0: f.s0 = v;return f;
      case 1: f.s1 = v;return f;
      case 2: f.s1 = v;return f;
      case 3: f.s1 = v;return f;
      case 4: f.s1 = v;return f;
      case 5: f.s1 = v;return f;
      case 6: f.s1 = v;return f;
      case 7: f.s1 = v;return f;
      default: return dummy;
    }
}
kernel void gemm_nn4W (const int M, const int N, const int K, const float ALPHA,
		 global float *restrict A, const int lda,
		 global float *restrict B, const int ldb,
		 global float *restrict C, const int ldc
		)
{
  int i, j, k;
  int wM = M/WRD;
  int wN = N/WRD;
  int wK = K/WRD;
  int wlda = lda/WRD;
  int wldb = ldb/WRD;
  int wldc = ldc/WRD;
  unsigned char countn=0;
  for (i = 0; i < wM; ++i) {
    for (j = 0; j < wN; j++) {
      float2 Cx = vload2((i*wldc+j), B);
      float Cn=sel_float2(Cx, countn);
	  for (k = 0; k < wK; ++k) {
        float2 Ax = vload2((i*wlda+k), A);
        float2 Bx = vload2((k*wldb+j), B);
        Cn+=dot2(Bx,Ax);
	  }
      Cx = set_float2(Cx,countn,Cn); // slow
      set_float2(Cx,countn,Cn); // fast
      if (countn++ == 2) countn = 0;
      vstore2(Cx,(i*wldc+j),C);
	}
  }
}
