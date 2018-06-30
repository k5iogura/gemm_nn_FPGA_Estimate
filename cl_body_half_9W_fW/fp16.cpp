#include <string.h>
#include "fp16.h"
#include "OpenEXR/half.h"

//
// float  : IEEE 754
// fp16   : unsigned short
//
//
fp16 f2h(float f){
    half h = f;
    fp16 p;
    memcpy(&p,&h,2);
    return   p;
}

float h2f(fp16 p){
    half h;
    memcpy(&h,&p,2);
    float b = h;
    return    b;
}

