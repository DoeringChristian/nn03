#include "layer_parallel.h"

struct layer_parallel *layer_parallel_init(struct layer_parallel *dst, size_t in_size, size_t out_size){
    dst->interface.ops = &layer_parallel_ops;
    dst->interface.in = NULL;
    dst->interface.out = NULL;
    dst->interface.in_grad = NULL;
    dst->interface.out_grad = NULL;
    dst->interface.in_size = out_size;
    dst->interface.out_size = out_size;
    dst->func = layer_parallel_sigmoid;
    dst->func_diff = layer_parallel_sigmoid_diff;
    return dst;
}
void layer_parallel_free(struct layer *dst){
    return;
}

int layer_parallel_forward(struct layer *ctx){
    struct layer_parallel *lp = (struct layer_parallel *)ctx;
    size_t i;

    if(ctx->in == NULL || ctx->out == NULL || lp->func == NULL)
        return 0;

    for(i = 0;i < ctx->in_size;i++)
        ctx->out[i] = lp->func(ctx->in[i]);

    return 1;
}
int layer_parallel_backward(struct layer *ctx){
    struct layer_parallel *lp = (struct layer_parallel *)ctx;
    size_t i;

    if(ctx->in == NULL || ctx->in_grad == NULL || ctx->out_grad == NULL || lp->func_diff == NULL)
        return 0;

    for(i = 0;i < ctx->in_size;i++)
        ctx->in_grad[i] = lp->func_diff(ctx->in[i]) * ctx->out_grad[i];

    return 1;
}
int layer_parallel_update(struct layer *ctx, float s){
    // do nothing since it has no weights.
    return 1;
}

float layer_parallel_sigmoid(float x){
    return 1/(1 + powf(M_E, -x));
}
float layer_parallel_sigmoid_diff(float x){
    float a = powf(M_E, -x);
    return a/((1+a) * (1+a));
}
