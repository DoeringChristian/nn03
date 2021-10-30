#include "network.h"

struct network *network_init(struct network *dst, const size_t *sizes, size_t depth){
    dst->depth = depth;
    dst->sizes = malloc(sizeof(size_t) * depth);
    dst->vecs = malloc(sizeof(float *) * depth);
    dst->vecs_grad = malloc(sizeof(float *) * depth);
#if 0
    dst->layers = malloc(sizeof(struct layer *) * depth - 1);
#endif

    size_t i;
    for(i = 0;i < depth;i++){
        dst->sizes[i] = sizes[i];
        dst->vecs[i] = vecn_new(sizes[i]);
        dst->vecs_grad[i] = vecn_new(sizes[i]);
    }
    return dst;
}
void network_free(struct network *dst){
    size_t i = 0;
    for(i = 0;i < dst->depth;i++){
        vecn_free(dst->vecs[i]);
        vecn_free(dst->vecs_grad[i]);
    }
    free(dst->sizes);
    dst->sizes = NULL;
    free(dst->vecs);
    dst->vecs = NULL;
    free(dst->vecs_grad);
    dst->vecs_grad = NULL;
#if 0
    free(dst->layers);
    dst->layers = NULL;
#endif
}
