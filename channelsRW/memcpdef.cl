#pragma OPENCL EXTENSION cl_altera_channels : enable
#pragma OPENCL EXTENSION cl_intel_channels : enable
#define WRD (32)
#define TYPE floatArray
typedef struct {
    float s[WRD];
} floatArray;

channel TYPE CHIN __attribute__((depth(0)));
kernel 
void mem_read(const int N, global TYPE *restrict src){
    int i=0;
    for(i=0;i<N/WRD;i++)
        write_channel_intel(CHIN,src[i]);
}

kernel 
void mem_write(const int N, global TYPE *restrict dst){
    int i=0;
    for(i=0;i<N/WRD;i++)
        dst[i] = read_channel_intel(CHIN);
}

