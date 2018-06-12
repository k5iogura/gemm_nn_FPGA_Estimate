*# gemm_nn_FPGA_Estimate*
# Simply c++ main program to use OpenCL I/F to invoke Kernel on Altera FPGA(DE0Nano).  
# And, GEMM_nn OpenCL kernel code from src/gemm.c Darknet.   

1. For CUDA
make -f Makefile.cuda
./gemm_cuda

2. For EMULATOR
make -f Makefile.emu
./gemm_emu

3. For FPGA
# cross compile
make -f Makefile.fpga
make -f Makefile.fpga gemm_fpga.aocx

# trasfer gemm_fpga and gemm_fpga.aocx onto FPGA Device
source init_opencl.sh
./gemm_fpga

