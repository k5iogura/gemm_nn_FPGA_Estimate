#pragma OPENCL EXTENSION cl_altera_channels : enable
#pragma OPENCL EXTENSION cl_intel_channels : enable
#define W (224)
#define H (160)
#define CIN1 (3)
#define CGO1 (16)
#define KSZ1 (3)
#define PAD1 (1)
#define SRD1 (1)
channel float CHIN1[CIN1] __attribute__((depth(0)));
channel float CHIN2[CGO1] __attribute__((depth(0)));
kernel 
void mem_read(const int N, global float *restrict src){
//printf("mem_read\t");
    int i,j;
    for(j=0;j<W*H;j++){
        #pragma unroll
        for(i=0;i<3;i++){
            write_channel_intel(CHIN1[i],src[j + i*W*H]);
        }
    }
}

float DOT9(constant float *A, float B[]){
    int i;
    float C=0;
    #pragma unroll
    for(i=0;i<KSZ1*KSZ1;i++)
        C+= A[i] * B[i];
    return C;
}

constant float krn[9*3] = {
    1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0,
    1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0,
    1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0
};
kernel
void conv(){
    float sr[CIN1][2*W+3];
    int i,j,k,g,l;

    for(i=0;i<CIN1;i++)
        for (j=0;j<2*W+3;j++)
            sr[i][j]=0;

    for(j=0;j<H*W;j++){
        #pragma unroll
        for(i=0;i<CIN1;i++){
            for (k=2*W+3-1;k>0;k--)
                sr[i][k] = sr[i][k-1];
            sr[i][2*W+3-1] = read_channel_intel(CHIN1[i]);
        }
        #pragma unroll
        for(g=0;g<CGO1;g++){
            float Cx=0;
            for(i=0;i<CIN1;i++){
                float pch[9] = {sr[i][    0], sr[i][    1], sr[i][    2],
                                sr[i][  W+0], sr[i][  W+1], sr[i][  W+2],
                                sr[i][2*W+0], sr[i][2*W+1], sr[i][2*W+2]};
                Cx += DOT9( krn+i*9 , pch );
            }
            write_channel_intel(CHIN2[g],Cx);
        }
    }
}

kernel 
void mem_write(const int N, global float *restrict dst){
//printf("mem_write\t");
    int i,j;
    for(j=0;j<W*H;j++){
        #pragma unroll
        for(i=0;i<CGO1;i++){
            dst[j+ i*W*H] = read_channel_intel(CHIN2[i]);
        }
    }
}

