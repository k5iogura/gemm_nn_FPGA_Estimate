#!/bin/bash

if [ -e ~/bin/src.opencl ];then
echo source src.opencl
. src.opencl
fi
echo BOARD $BOARD
echo ALTERAOCLSDKROOT $ALTERAOCLSDKROOT
echo AOCL_BOARD_PACKAGE_ROOT $AOCL_BOARD_PACKAGE_ROOT

echo INTO SHLVL=$SHLVL
export SHLVL=`expr $SHLVL - 1`
/tools/intelFPGA/18.0/embedded/embedded_command_shell.sh

