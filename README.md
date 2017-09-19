*# gemm_nn_FPGA_Estimate*
# Simply c++ main program to use OpenCL I/F to invoke Kernel on Altera FPGA(DE0Nano).  
# And, GEMM_nn OpenCL kernel code from src/gemm.c Darknet.   

1.Makefiles
Makefile.x86:to compile gemm1.cpp for Windows  
Makefile.arm:to compile gemm1.cpp for SoC(ARM) in FPGA  
2.gemm1.cpp  
  main program  
3.gemm1.cl  
  OpenCL kernel for GEMM_nn operation.  
  export AOCL_BOARD_PACKAGE_ROOT=N:\\win_shared\\DE0_NANO\\OpenCL\\opencl_soc_bsp-de0_nano_with_display\\c5soc  
  aoc --list-boards  
  echo 'type to compile opencl: aoc gemm1.cl --board de0_nano_sharedonly_with_spi_tft'  
  
