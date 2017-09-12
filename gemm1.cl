float gemm_4W (float A, float Bx, float Cx){ return(A * Bx + Cx); }
__kernel
void gemm_nn4W (const int M, const int N, const int K, const float ALPHA,
		 global float *restrict A, const int lda,
		 global float *restrict B, const int ldb,
		 global float *restrict C, const int ldc
		)
{
  int i, j, k, m;
  for (i = 0; i < M; ++i) {
	for (k = 0; k < K; ++k) {
	  float A_PART = A[i * lda + k];
	  for (j = 0; j < N; j+=4) {
	    float B0, B1, B2, B3, C0, C1, C2, C3;
		B0 = B[k * ldb + j + 0];
		B1 = B[k * ldb + j + 1];
		B2 = B[k * ldb + j + 2];
		B3 = B[k * ldb + j + 3];
		C0 = C[i * ldc + j + 0];
		C1 = C[i * ldc + j + 1];
		C2 = C[i * ldc + j + 2];
		C3 = C[i * ldc + j + 3];
		C[i * ldc + j + 0] = gemm_4W(A_PART , B0 , C0);
		C[i * ldc + j + 1] = gemm_4W(A_PART , B1 , C1);
		C[i * ldc + j + 2] = gemm_4W(A_PART , B2 , C2);
		C[i * ldc + j + 3] = gemm_4W(A_PART , B3 , C3);
	  }
	}
  }
}
/*__kernel
void gemm_nnA1 (const int M, const int N, const int K, const float ALPHA,
		 global float *restrict A, const int lda,
		 global float *restrict B, const int ldb,
		 global float *restrict C, const int ldc
		)
{
  int i, j, k;
  for (i = 0; i < M; ++i) {
	for (k = 0; k < K; ++k) {
	  float A_PART = A[i * lda + k];
	  for (j = 0; j < N; ++j) {
		C[i * ldc + j] += A_PART * B[k * ldb + j];
	  }
	}
  }
}*/
//const float WNorm[2304];	// Not Allow Extern variable
/*#define ConstM 20736
__kernel
void gemm_nnWB (const int M, const int N, const int K, const float ALPHA,
		 global float *restrict A, const int lda,
		 global float *restrict B, const int ldb,
		 global float *restrict C, const int ldc
		)
{
  int i, j, k;
  //const float WNorm[20736];	// Not Assignable! Compile Error
  //const float WNorm[9216];	// Not Assignable! Compile Error
  //const float WNorm[2304];	//// Not Assignable! Compile Error
  //local float WNorm[ConstM]={0.0};
  //for (i = 0; i < ConstM; ++i) WNorm[i] = 3.14;
  for (i = 0; i < M; ++i) {
	for (k = 0; k < K; ++k) {
	  float A_PART = A[i * lda + k];
	  for (j = 0; j < N; ++j) {
	    float B_PART = B[k * ldb + j];
		C[i * ldc + j] += (A_PART>=0.0)?B_PART:-B_PART;
	  }
	}
  }
}*/
/*__kernel
void gemm_nn (const int M, const int N, const int K, const float ALPHA,
		 global float *restrict A, const int lda,
		 global float *restrict B, const int ldb,
		 global float *restrict C, const int ldc
		)
{
  int i, j, k;
  for (i = 0; i < M; ++i) {
	for (k = 0; k < K; ++k) {
	  float A_PART = ALPHA * A[i * lda + k];
	  for (j = 0; j < N; ++j) {
		C[i * ldc + j] += A_PART * B[k * ldb + j];
	  }
	}
  }
}
__kernel
void gemm_nnA1 (const int M, const int N, const int K, const float ALPHA,
		 global float *restrict A, const int lda,
		 global float *restrict B, const int ldb,
		 global float *restrict C, const int ldc
		)
{
  int i, j, k;
  float b;
  for (i = 0; i < M*K; ++i) {
     C[i%N]=A[i];
  }
  for (i = 0; i < K*N; ++i) {
     C[i%N]=B[i];
  }
  for (i = 0; i < M*N; ++i) {
     A[i%K]=C[i];
  }
}*/
