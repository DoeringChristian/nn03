#include "layer_mat.h"

struct layer_mat *layer_mat_init(struct layer_mat *dst, struct node_ctx in_ctx, struct node_ctx out_ctx){
    dst->interface.ops = &layer_mat_ops;
    dst->interface.in_ctx = in_ctx;
    dst->interface.out_ctx = out_ctx;
    dst->mat = matnm_new(in_ctx.n, out_ctx.n);
    dst->mat_diff = matnm_new(in_ctx.n, out_ctx.n);
    return dst;
}

void layer_mat_free(struct layer *dst){
    struct layer_mat *lm = (struct layer_mat *)dst;
    matnm_free(lm->mat);
    matnm_free(lm->mat_diff);
    lm->mat = NULL;
    lm->mat_diff = NULL;
}

int layer_mat_forward(struct layer *ctx){
    struct layer_mat *lm = (struct layer_mat *)ctx;

    if(!node_ctx_valid(&ctx->in_ctx) || !node_ctx_valid(&ctx->out_ctx))
        return 0;

    matnm_mult(node_ctx_state(&ctx->out_ctx), lm->mat, node_ctx_state(&ctx->in_ctx), ctx->in_ctx.n, ctx->out_ctx.n, 1);
    return 1;
}
int layer_mat_backward(struct layer *ctx){
    struct layer_mat *lm = (struct layer_mat *)ctx;

    if(!node_ctx_valid(&ctx->in_ctx) || !node_ctx_valid(&ctx->out_ctx))
        return 0;

    matnm_mult(ctx->in_ctx.node->state, ctx->out_ctx.node->grad, lm->mat, ctx->out_ctx.n, 1, ctx->in_ctx.n);
    matnm_mult(lm->mat_diff, ctx->in_ctx.node->state, ctx->out_ctx.node->grad, 1, ctx->in_ctx.n, ctx->out_ctx.n);
    return 1;
}
int layer_mat_update(struct layer *ctx, float s){
    struct layer_mat *lm = (struct layer_mat *)ctx;
    
    matnm_scale(lm->mat_diff, lm->mat_diff, s, ctx->in_ctx.n, ctx->out_ctx.n);
    matnm_add(lm->mat, lm->mat, lm->mat_diff, ctx->in_ctx.n, ctx->out_ctx.n);
    return 1;
}
