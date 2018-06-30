#ifndef __EASY_HALF_H__
#define __EASY_HALF_H__
typedef unsigned short fp16;

typedef union {
    unsigned int n;
    float f;
} float_convert;

fp16 f2h(float f);
float h2f(fp16 a);
#endif
