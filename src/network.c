#include "network.h"

struct network *network_init(struct network *dst, const size_t *sizes, const enum layer_type *types, size_t depth){
    dst->depth = depth;
    dst->sizes = malloc(sizeof(size_t) * depth);
    dst->vecs = malloc(sizeof(float *) * depth);
    dst->vecs_grad = malloc(sizeof(float *) * depth);
    dst->layers = malloc(sizeof(struct layer *) * depth - 1);


    size_t i;
    for(i = 0;i < depth;i++){
        dst->sizes[i] = sizes[i];
        dst->vecs[i] = vecn_new(sizes[i]);
        dst->vecs_grad[i] = vecn_new(sizes[i]);
    }

    for(i = 0;i < depth-1;i++){
        switch(types[i]){
        case LAYER_TYPE_ERR:
            dst->layers[i] = malloc(sizeof(struct layer_err));
            layer_err_init((struct layer_err *)dst->layers[i], sizes[i], sizes[i+1]);
            break;
        case LAYER_TYPE_MAT:
            dst->layers[i] = malloc(sizeof(struct layer_mat));
            layer_mat_init((struct layer_mat *)dst->layers[i], sizes[i], sizes[i+1]);
            break;
        case LAYER_TYPE_PARALLEL:
            dst->layers[i] = malloc(sizeof(struct layer_parallel));
            layer_parallel_init((struct layer_parallel *)dst->layers[i], sizes[i], sizes[i+1]);
            break;
        }
        dst->layers[i]->in = dst->vecs[i];
        dst->layers[i]->out = dst->vecs[i+1];
        dst->layers[i]->in_grad = dst->vecs_grad[i];
        dst->layers[i]->out_grad = dst->vecs_grad[i+1];
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
    free(dst->layers);
    dst->layers = NULL;
}
