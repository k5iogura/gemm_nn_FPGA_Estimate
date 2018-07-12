#pragma OPENCL EXTENSION cl_altera_channels : enable
#pragma OPENCL EXTENSION cl_intel_channels : enable
channel uint CHIN __attribute__((depth(0)));
__attribute__((reqd_work_group_size(16,1,1)))
kernel 
void mem_read(const int N, global uint *restrict src){
    int i=0;
    size_t gid = get_global_id(0);
    size_t siz = get_global_size(0);

    for(i=0+gid;i<N+gid-siz+1;i+=siz)
        write_channel_intel(CHIN,src[i]);
}

kernel 
void mem_write(const int N, global uint *restrict dst){
    int i=0;
    for(i=0;i<N;i++)
        dst[i] = read_channel_intel(CHIN);
}

