#include "layer_mat.h"
#include <stdio.h>

static inline int matn_print(const float *src, size_t n){
    for(size_t i = 0;i < n;i++){
        printf("|");
        for(size_t j = 0;j < n-1;j++)
            printf("%f, ", matn_get(src, j, i, n));
        printf("%f", matn_get(src, n-1, i, n));
        printf("|\n");
    }
    return 1;
}

static inline int matnm_print(const float *src, size_t n, size_t m){
    for(size_t i = 0;i < m;i++){
        printf("|");
        for(size_t j = 0;j < n-1;j++)
            printf("%f, ", matnm_get(src, j, i, n));
        printf("%f", matnm_get(src, n-1, i, n));
        printf("|\n");
    }
    return 1;
}

static inline int vecn_print(const float *src, size_t n){
    printf("(");
    for(size_t i = 0;i < n-1;i++)
        printf("%f, ", src[i]);
    printf("%f)\n", src[n-1]);
    return 1;
}

float *vecn_rand(float *dst, size_t n){
    size_t i;
    for(i = 0;i < n;i++)
        dst[i] = (float)rand() / (float)RAND_MAX;
    return dst;
}

int main(){
    srand(1);



    return 0;
}

