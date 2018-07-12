#pragma OPENCL EXTENSION cl_altera_channels : enable
#pragma OPENCL EXTENSION cl_intel_channels : enable
#define W (224)
#define H (160)
typedef struct { float s[ 3]; } Float3;
typedef struct { float s[16]; } Float16;
channel Float3  CHIN1 __attribute__((depth(0)));
channel Float16 CHIN2 __attribute__((depth(0)));
kernel 
void mem_read(const int N, global Float3 *restrict src){
    int i=0;
    for(i=0;i<W*H/3;i++)
        write_channel_intel(CHIN1, src[i]);
}

float sum9(Float3* a){
    int i;
    float b=0;
    #pragma unroll
    for(i=0;i<9;i++)
        b+=a[i].s[0] + a[i].s[1] + a[i].s[2];
    return b;
}
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
kernel
void conv(){
    Float3 sr[2*W+3];
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

//    for (i=0;i<2*W+3;i++)
//        sr[i]=0;

    for(j=0;j<H*W/3;j++){
//        #pragma unroll
        for (i=2*W+3-1;i>0;i--)
            sr[i] = sr[i-1];
        sr[2*W+3-1] = read_channel_intel(CHIN1);

        Float16 Cx;
        for(l=0;l<16;l++){
            Float3 pch[] = {sr[    0], sr[    1], sr[    2],
                            sr[  W+0], sr[  W+1], sr[  W+2],
                            sr[2*W+0], sr[2*W+1], sr[2*W+2]};
            Cx.s[l]+= sum9(pch);
        }
        write_channel_intel(CHIN2,Cx);
    }
}

kernel 
void mem_write(const int N, global Float16 *restrict dst){
    int i;
    for(i=0;i<W*H/3;i++)
        dst[i] = read_channel_intel(CHIN2);
}

