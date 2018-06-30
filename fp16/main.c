#include <stdio.h>
#include "fp16.h"

main(){

    int i;
    for(i=0;i<10000;i+=10){
        float a=i/10000.;
        fp16 xxx = f2h(a);
        float yyy = h2f(xxx);
        printf("a/yyy = %f/%f (%f)\n",a,yyy,a-yyy);
    }
}
