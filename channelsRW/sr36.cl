#pragma OPENCL EXTENSION cl_altera_channels : enable
#pragma OPENCL EXTENSION cl_intel_channels : enable
#define W (224)
#define H (160)
channel float CHIN1[ 3] __attribute__((depth(0)));
channel float CHIN2[16] __attribute__((depth(0)));
kernel 
void mem_read(const int N, global float *restrict src){
//printf("mem_read\t");
    int i=0;
    for(i=0;i<W*H;i++){
        write_channel_intel(CHIN1[0],src[i        ]);
        write_channel_intel(CHIN1[1],src[i +   W*H]);
        write_channel_intel(CHIN1[2],src[i + 2*W*H]);
    }
}

float sum9(float* a){
    int i;
    float b=0;
    #pragma unroll
    for(i=0;i<9;i++) b+=a[i];
    return b;
}
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
kernel
void conv(){
    printf("conv\t");
    float sr[3][2*W+3];
    int i,j,k,l;
    float wei[16][9]={
        {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}, 
        {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}, 
        {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}, 
        {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}, 
        {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}, 
        {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}, 
        {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}, 
        {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}, 
        {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}, 
        {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}, 
        {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}, 
        {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}, 
        {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}, 
        {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}, 
        {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}, 
        {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}, 
    };

    for(k=0;k<3;k++)
        for (i=0;i<2*W+3;i++)
            sr[0][i]=0;

    for(j=0;j<H*W;j++){
        #pragma unroll
        for(k=0;k<3;k++)
            for (i=2*W+3-1;i>0;i--)
                sr[k][i] = sr[k][i-1];

        float Cx[16];
        #pragma unroll
        for(k=0;k<3;k++)
            sr[k][2*W+3-1] = read_channel_intel(CHIN1[k]);
        for(l=0;l<16;l++)
            for(k=0;k<3;k++){
                float pch[9] = {sr[k][    0], sr[k][    1], sr[k][    2],
                                sr[k][  W+0], sr[k][  W+1], sr[k][  W+2],
                                sr[k][2*W+0], sr[k][2*W+1], sr[k][2*W+2]};
                Cx[l]+= sum9(pch);
            }
        #pragma unroll
        for(l=0;l<16;l++)
            write_channel_intel(CHIN2[l],Cx[l]);
    }
}

kernel 
void mem_write(const int N, global float *restrict dst){
    int i,k;
    for(i=0;i<W*H;i++){
        #pragma unroll
        for(k=0;k<16;k++)
            dst[i + k*W*H] = read_channel_intel(CHIN2[k]);
    }
}

