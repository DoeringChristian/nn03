#include "layer_err.h"

struct layer_err *layer_err_init(struct layer_err *dst, size_t in_size, size_t out_size){
    dst->interface.ops = &layer_err_ops;
    dst->interface.in = NULL;
    dst->interface.in_grad = NULL;
    dst->interface.out = NULL;
    dst->interface.out_grad = NULL;
    dst->interface.in_size = in_size;
    dst->interface.out_size = 1;
    dst->ref = NULL;
    return dst;
}
void layer_err_free(struct layer *dst){
    return;
}

int layer_err_forward(struct layer *ctx){
    struct layer_err *le = (struct layer_err *)ctx;
    
    if(ctx->in == NULL || ctx->out == NULL || le->ref == NULL)
        return 0;

    ctx->out[0] = vecn_dist_squared(le->ref, ctx->in, ctx->in_size);

    return 1;
}
int layer_err_backward(struct layer *ctx){
    struct layer_err *le = (struct layer_err *)ctx;

    if(ctx->in == NULL || ctx->in_grad == NULL || le->ref == NULL)
        return 0;

    size_t i;
    for(i = 0;i < ctx->in_size;i++)
        ctx->in_grad[i] = 2*(le->ref[i] - ctx->in[i])*(-ctx->in[i]);

    return 1;
}
int layer_err_update(struct layer *ctx, float s){
    return 1;
}
