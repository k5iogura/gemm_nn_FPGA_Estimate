#include "fp16.h"

//
// float  : IEEE 754
// fp16   : unsigned short
//
//
fp16 f2h(float a){
    float_convert c;

    c.f=a;
    unsigned int n = c.n;

    fp16 sign_bit = (n>>16)&0x8000;
    fp16 exponent = (((n >> 23) - 127 + 15) & 0x1f) << 10;
    fp16 fraction;
    if((n >> (23-10-1))& 0x1)
        fraction =((n >> (23-10))+0x1) & 0x3ff;
    else
        fraction = (n >> (23-10)) & 0x3ff;
    fp16 hf = sign_bit | exponent | fraction;
    return hf;
}

float h2f(fp16 a){
    float_convert c;
    unsigned int sign_bit = ( a & 0x8000 ) << 16;
    unsigned int exponent = ((((a >> 10) & 0x1f) - 15 + 127) & 0xff) << 23;
    unsigned int fraction = (a & 0x3ff) << (23 - 10);
    c.n = sign_bit | exponent | fraction;
    return c.f;
}

