#include "layer_parallel.h"

struct layer_parallel *layer_parallel_init(struct layer_parallel *dst, struct node_ctx in_ctx, struct node_ctx out_ctx){
    dst->interface.ops = &layer_parallel_ops;
    dst->interface.in_ctx = in_ctx;
    dst->interface.out_ctx = out_ctx;
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

    if(!node_ctx_valid(&ctx->in_ctx) || !node_ctx_valid(&ctx->out_ctx))
        return 0;

    for(i = 0;i < ctx->in_ctx.n;i++)
        ctx->out_ctx.node->state[i] = lp->func(ctx->in_ctx.node->state[i]);

    return 1;
}
int layer_parallel_backward(struct layer *ctx){
    struct layer_parallel *lp = (struct layer_parallel *)ctx;
    size_t i;

    if(!node_ctx_valid(&ctx->in_ctx) || !node_ctx_valid(&ctx->out_ctx))
        return 0;

    for(i = 0;i < ctx->in_ctx.n;i++)
        ctx->in_ctx.node->grad[i] = lp->func_diff(ctx->in_ctx.node->state[i]) * ctx->out_ctx.node->grad[i];

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
