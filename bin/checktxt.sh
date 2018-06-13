#!/usr/bin/env bash
count=0
found=0
for i in `cat $1`;
do
export TXT=`echo $i | sed 's/\.jpg/.txt/;'`
#echo check $i $TXT
if [ ! -r $TXT ];
then
    echo not found $TXT
else
    found=$((count+1))
fi
count=$((count+1))
done

if [ $count>=1 ];
then
    echo found/count $found/$count
fi
