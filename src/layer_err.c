#include "layer_err.h"

struct layer_err *layer_err_init(struct layer_err *dst, struct node_ctx in_ctx, struct node_ctx out_ctx){
    dst->interface.ops = &layer_err_ops;
    dst->interface.in_ctx = in_ctx;
    dst->interface.out_ctx = out_ctx;
    dst->ref = NULL;
}
void layer_err_free(struct layer *dst){
    return;
}

int layer_err_forward(struct layer *ctx){
    struct layer_err *le = (struct layer_err *)ctx;
    
    if(!node_ctx_valid(&ctx->in_ctx) || !node_ctx_valid(&ctx->out_ctx) || le->ref == NULL)
        return 0;

    ctx->out_ctx.node->state[0] = vecn_dist_squared(le->ref, ctx->in_ctx.node->state, ctx->in_ctx.n);

    return 1;
}
int layer_err_backward(struct layer *ctx){
    struct layer_err *le = (struct layer_err *)ctx;

    if(ctx->in_ctx.node->state == NULL || ctx->in_ctx.node->grad == NULL || le->ref == NULL)
        return 0;

    size_t i;
    for(i = 0;i < ctx->in_ctx.n;i++)
        ctx->in_ctx.node->grad[i] = 2*(le->ref[i] - ctx->in_ctx.node->state[i])*(-ctx->in_ctx.node->state[i]);

    return 1;
}
int layer_err_update(struct layer *ctx, float s){
    return 1;
}
