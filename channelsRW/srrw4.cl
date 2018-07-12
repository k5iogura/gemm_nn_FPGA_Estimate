#pragma OPENCL EXTENSION cl_altera_channels : enable
#pragma OPENCL EXTENSION cl_intel_channels : enable
#define W (224)
#define H (160)
channel float CHIN1[3] __attribute__((depth(0)));
channel float CHIN2[3] __attribute__((depth(0)));
channel float CHIN3[3] __attribute__((depth(0)));
channel float CHIN4[3] __attribute__((depth(0)));
channel float CHIN5[3] __attribute__((depth(0)));
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

kernel
void conv(){
//printf("conv\t");
    float sr[3][2*W+3];
    int i,j;

    for (i=0;i<2*W+3;i++)
        sr[0][i]=0;
    for (i=0;i<2*W+3;i++)
        sr[1][i]=0;
    for (i=0;i<2*W+3;i++)
        sr[2][i]=0;

    for(j=0;j<H*W;j++){
        //#pragma unroll
        for (i=2*W+3-1;i>0;i--)
            sr[0][i] = sr[0][i-1];
        //#pragma unroll
        for (i=2*W+3-1;i>0;i--)
            sr[1][i] = sr[1][i-1];
        //#pragma unroll
        for (i=2*W+3-1;i>0;i--)
            sr[2][i] = sr[2][i-1];

        sr[0][2*W+3-1] = read_channel_intel(CHIN1[0]);
        sr[1][2*W+3-1] = read_channel_intel(CHIN1[1]);
        sr[2][2*W+3-1] = read_channel_intel(CHIN1[2]);
        write_channel_intel(CHIN2[0],sr[0][0]);
        write_channel_intel(CHIN2[1],sr[1][0]);
        write_channel_intel(CHIN2[2],sr[2][0]);
    }
}

kernel
void conv2(){
//printf("conv\t");
    float sr[3][2*W+3];
    int i,j;

    for (i=0;i<2*W+3;i++)
        sr[0][i]=0;
    for (i=0;i<2*W+3;i++)
        sr[1][i]=0;
    for (i=0;i<2*W+3;i++)
        sr[2][i]=0;

    for(j=0;j<H*W;j++){
        //#pragma unroll
        for (i=2*W+3-1;i>0;i--)
            sr[0][i] = sr[0][i-1];
        //#pragma unroll
        for (i=2*W+3-1;i>0;i--)
            sr[1][i] = sr[1][i-1];
        //#pragma unroll
        for (i=2*W+3-1;i>0;i--)
            sr[2][i] = sr[2][i-1];

        sr[0][2*W+3-1] = read_channel_intel(CHIN2[0]);
        sr[1][2*W+3-1] = read_channel_intel(CHIN2[1]);
        sr[2][2*W+3-1] = read_channel_intel(CHIN2[2]);
        write_channel_intel(CHIN3[0],sr[0][0]);
        write_channel_intel(CHIN3[1],sr[1][0]);
        write_channel_intel(CHIN3[2],sr[2][0]);
    }
}

kernel
void conv3(){
//printf("conv\t");
    float sr[3][2*W+3];
    int i,j;

    for (i=0;i<2*W+3;i++)
        sr[0][i]=0;
    for (i=0;i<2*W+3;i++)
        sr[1][i]=0;
    for (i=0;i<2*W+3;i++)
        sr[2][i]=0;

    for(j=0;j<H*W;j++){
        //#pragma unroll
        for (i=2*W+3-1;i>0;i--)
            sr[0][i] = sr[0][i-1];
        //#pragma unroll
        for (i=2*W+3-1;i>0;i--)
            sr[1][i] = sr[1][i-1];
        //#pragma unroll
        for (i=2*W+3-1;i>0;i--)
            sr[2][i] = sr[2][i-1];

        sr[0][2*W+3-1] = read_channel_intel(CHIN3[0]);
        sr[1][2*W+3-1] = read_channel_intel(CHIN3[1]);
        sr[2][2*W+3-1] = read_channel_intel(CHIN3[2]);
        write_channel_intel(CHIN4[0],sr[0][0]);
        write_channel_intel(CHIN4[1],sr[1][0]);
        write_channel_intel(CHIN4[2],sr[2][0]);
    }
}

kernel
void conv4(){
//printf("conv\t");
    float sr[3][2*W+3];
    int i,j;

    for (i=0;i<2*W+3;i++)
        sr[0][i]=0;
    for (i=0;i<2*W+3;i++)
        sr[1][i]=0;
    for (i=0;i<2*W+3;i++)
        sr[2][i]=0;

    for(j=0;j<H*W;j++){
        //#pragma unroll
        for (i=2*W+3-1;i>0;i--)
            sr[0][i] = sr[0][i-1];
        //#pragma unroll
        for (i=2*W+3-1;i>0;i--)
            sr[1][i] = sr[1][i-1];
        //#pragma unroll
        for (i=2*W+3-1;i>0;i--)
            sr[2][i] = sr[2][i-1];

        sr[0][2*W+3-1] = read_channel_intel(CHIN4[0]);
        sr[1][2*W+3-1] = read_channel_intel(CHIN4[1]);
        sr[2][2*W+3-1] = read_channel_intel(CHIN4[2]);
        write_channel_intel(CHIN5[0],sr[0][0]);
        write_channel_intel(CHIN5[1],sr[1][0]);
        write_channel_intel(CHIN5[2],sr[2][0]);
    }
}

kernel 
void mem_write(const int N, global float *restrict dst){
//printf("mem_write\t");
    int i=0;
    for(i=0;i<W*H;i++){
        dst[i       ] = read_channel_intel(CHIN5[0]);
        dst[i+   W*H] = read_channel_intel(CHIN5[1]);
        dst[i+ 2*W*H] = read_channel_intel(CHIN5[2]);
    }
}

