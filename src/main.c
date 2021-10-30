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

    struct layer_mat lm;
    layer_mat_init(&lm, 3, 3);
    lm.interface.in =  vecn_new(lm.interface.in_size);
    lm.interface.in_grad = vecn_new(lm.interface.in_size);
    lm.interface.out = vecn_new(lm.interface.out_size);
    lm.interface.out_grad = vecn_new(lm.interface.out_size);

    vecn_rand(lm.interface.in, lm.interface.in_size);
    vecn_rand(lm.interface.out, lm.interface.out_size);
    vecn_rand(lm.mat, lm.interface.in_size * lm.interface.out_size);

    printf("forward:\n");
    matnm_print(lm.interface.in, 1, lm.interface.in_size);
    printf("->\n");
    matnm_print(lm.mat, lm.interface.in_size, lm.interface.out_size);


    layer_forward((struct layer *)&lm);
    printf("->\n");
    matnm_print(lm.interface.out, 1, lm.interface.out_size);

    vecn_rand(lm.interface.out_grad, lm.interface.out_size);
    vecn_zero(lm.mat_diff, lm.interface.in_size * lm.interface.out_size);

    printf("backward:\n");
    printf("in:\n");
    matnm_print(lm.interface.in, 1, lm.interface.in_size);
    printf("\n");
    printf("outT:\n");
    matnm_print(lm.interface.out_grad, lm.interface.out_size, 1);
    printf("<-\n");
    
    layer_backward((struct layer *)&lm);
    printf("mat_diff:\n");
    matnm_print(lm.mat_diff, lm.interface.in_size, lm.interface.out_size);
    printf("\n");
    printf("in_grad:\n");
    matnm_print(lm.interface.in_grad, 1, lm.interface.in_size);
    


    return 0;
}

