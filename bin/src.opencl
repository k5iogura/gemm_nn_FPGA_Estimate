#!/bin/bash
if [ -z $QUARTUS_ROOTDIR ];then
    export QUARTUS_ROOTDIR=/opt/intelFPGA/18.0/quartus
fi
if [ -z $ALTERAOCLSDKROOT ];then
    export ALTERAOCLSDKROOT=/opt/intelFPGA/18.0/hld
fi
if [ -z $QUARTUS_64BIT ];then
    export QUARTUS_64BIT=1
    export PATH=$PATH:$QUARTUS_ROOTDIR/bin:/opt/intelFPGA/18.0/embedded/ds-5/bin:/opt/intelFPGA/18.0/embedded/ds-5/sw/gcc/bin:$ALTERAOCLSDKROOT/bin:$ALTERAOCLSDKROOT/linux64/bin:
    export LD_LIBRARY_PATH=$ALTERAOCLSDKROOT/linux64/lib
    #export LM_LICENSE_FILE=/home/ogura/Licese.dat
fi

if [ -z $BOARD ];then
    echo IN EMULATOR-MODE
    export AOCL_BOARD_PACKAGE_ROOT=/home/20076433/20076433/BSP/c5gt
    export BOARD=c5soc
    export CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1
    #2017.11.04 Reblanded CL_CONTEXT_EMULATOR_DEVICE_ALTERA to
    export CL_CONTEXT_EMULATOR_DEVICE_INTELFPGA=1
elif [ $BOARD == "de10_nano" ];then
    echo IN FPGA-MODE ON $BOARD
    export AOCL_BOARD_PACKAGE_ROOT=/home/`whoami`/`whoami`/BSP/$BOARD
    export BOARD=de10_nano
fi
echo AOCL_BOARD_PACKAGE_ROOT $AOCL_BOARD_PACKAGE_ROOT

#2017.11.04 Reblanded ALTERAOCLSDKROOT to
export INTELFPGAOCLSDKROOT=$ALTERAOCLSDKROOT
export LD_LIBRARY_PATH=$INTELFPGAOCLSDKROOT/host/linux64/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$AOCL_BOARD_PACKAGE_ROOT/linux64/lib:$LD_LIBRARY_PATH
export QUARTUS_ROOTDIR_OVERRIDE=$QUARTUS_ROOTDIR
#export TMPDIR=/home/`whoami`/tmpdir

