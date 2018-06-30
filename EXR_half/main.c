#include <stdio.h>
#include "OpenEXR/half.h"

main(){

    int i;
    for(i=0;i<10000;i+=10){
        float a=i/10000.;
        half xxx = a;
        float yyy = xxx;
        printf("a/yyy = %f/%f (%f)\n",a,yyy,a-yyy);
    }
}
