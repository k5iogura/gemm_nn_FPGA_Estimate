#pragma OPENCL EXTENSION cl_altera_channels : enable
#pragma OPENCL EXTENSION cl_intel_channels : enable
#define W (224)
#define H (160)
channel float3  CHIN1 __attribute__((depth(0)));
channel float16 CHIN2 __attribute__((depth(0)));
kernel 
void mem_read(const int N, global float3 *restrict src){
//printf("mem_read\t");
    int i=0;
    for(i=0;i<W*H/3;i++){
        write_channel_intel(CHIN1, src[i]);
    }
}

float sum9(float3* a){
    int i;
    float b=0;
    #pragma unroll
    for(i=0;i<9;i++)
        b+=a[i].s0 + a[i].s1 + a[i].s2;
    return b;
}
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
kernel
void conv(){
//printf("conv\t");
    float3 sr[2*W+3];
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

    for (i=0;i<2*W+3;i++)
        sr[i]=0;

    for(j=0;j<H*W/3;j++){
//        #pragma unroll
        for (i=2*W+3-1;i>0;i--)
            sr[i] = sr[i-1];
        sr[2*W+3-1] = read_channel_intel(CHIN1);

        float Cx[16];
        for(l=0;l<16;l++){
            float3 pch[] = {sr[    0], sr[    1], sr[    2],
                            sr[  W+0], sr[  W+1], sr[  W+2],
                            sr[2*W+0], sr[2*W+1], sr[2*W+2]};
            Cx[l]+= sum9(pch);
        }
        float16 Cxf = (float16)(
            Cx[ 0], Cx[ 1], Cx[ 2], Cx[ 3], 
            Cx[ 4], Cx[ 5], Cx[ 6], Cx[ 7], 
            Cx[ 8], Cx[ 9], Cx[10], Cx[11], 
            Cx[12], Cx[13], Cx[14], Cx[15] 
        );
        write_channel_intel(CHIN2,Cxf);
    }
}

kernel 
void mem_write(const int N, global float16 *restrict dst){
    int i;
    for(i=0;i<W*H/3;i++)
        dst[i] = read_channel_intel(CHIN2);
}

