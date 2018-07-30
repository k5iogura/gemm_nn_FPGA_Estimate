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

constant Float3 w0[] = { {{.1,.1,.1}}, {{.1,.1,.1}}, {{.1,.1,.1}}, {{.1,.1,.1}}, {{.1,.1,.1}}, {{.1,.1,.1}}, {{.1,.1,.1}}, {{.1,.1,.1}}, {{.1,.1,.1}}, };
constant Float3 w1[] = { {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, };
constant Float3 w2[] = { {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, };
constant Float3 w3[] = { {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, };
constant Float3 w4[] = { {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, };
constant Float3 w5[] = { {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, };
constant Float3 w6[] = { {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, };
constant Float3 w7[] = { {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, };
constant Float3 w8[] = { {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, };
constant Float3 w9[] = { {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, };
constant Float3 wa[] = { {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, };
constant Float3 wb[] = { {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, };
constant Float3 wc[] = { {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, };
constant Float3 wd[] = { {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, };
constant Float3 we[] = { {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, };
constant Float3 wf[] = { {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, {{.2,.2,.2}}, };
float sum9(Float3* a, constant Float3* w){
    int i;
    float b=0;
    #pragma unroll
    for(i=0;i<9;i++)
        b+=
            a[i].s[0] * w[i].s[0] +
            a[i].s[1] * w[i].s[1] +
            a[i].s[2] * w[i].s[2];
    return b;
}
__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
kernel
void conv(){
    Float3 sr[2*W+3];
    int i,j,k,l;
    constant Float3 *WW[16]={
        w0,w1,w2,w3,w4,w5,w6,w7,w8,w9,wa,wb,wc,wd,we,wf
    };
//    for (i=0;i<2*W+3;i++)
//        sr[i]=0;

    for(j=0;j<H*W/3;j++){
        for (i=2*W+3-1;i>0;i--)
            sr[i] = sr[i-1];
        sr[2*W+3-1] = read_channel_intel(CHIN1);

        Float3 pch[] = {sr[    0], sr[    1], sr[    2],
                        sr[  W+0], sr[  W+1], sr[  W+2],
                        sr[2*W+0], sr[2*W+1], sr[2*W+2]};
        Float16 Cx;
        #pragma unroll
        for(l=0;l<16;l++)
            Cx.s[l] = sum9(pch,WW[l]);
        write_channel_intel(CHIN2,Cx);
    }
}

kernel 
void mem_write(const int N, global Float16 *restrict dst){
    int i;
    for(i=0;i<W*H/3;i++)
        dst[i] = read_channel_intel(CHIN2);
}
