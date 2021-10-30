#include "layer_mat.h"

struct layer_mat *layer_mat_init(struct layer_mat *dst, size_t in_size, size_t out_size){
    dst->interface.ops = &layer_mat_ops;
    dst->interface.in_size = in_size;
    dst->interface.out_size = out_size;
    dst->mat = matnm_new(in_size, out_size);
    dst->mat_diff = matnm_new(in_size, out_size);
    dst->interface.in = NULL;
    dst->interface.out = NULL;
    dst->interface.in_grad = NULL;
    dst->interface.out_grad = NULL;
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

    if(ctx->in == NULL || ctx->out == NULL)
        return 0;

    matnm_mult(ctx->out, lm->mat, ctx->in, lm->interface.in_size + 1, lm->interface.out_size, 1);
    return 1;
}
int layer_mat_backward(struct layer *ctx){
    struct layer_mat *lm = (struct layer_mat *)ctx;

    if(ctx->in == NULL || ctx->out_grad == NULL || ctx->in_grad == NULL)
        return 0;

    matnm_mult(ctx->in_grad, ctx->out_grad, lm->mat, ctx->out_size, 1, ctx->in_size);
    matnm_mult(lm->mat_diff, ctx->in, ctx->out_grad, 1, ctx->in_size, ctx->out_size);
    return 1;
}
int layer_mat_update(struct layer *ctx, float s){
    struct layer_mat *lm = (struct layer_mat *)ctx;
    
    matnm_scale(lm->mat_diff, lm->mat_diff, s, ctx->in_size, ctx->out_size);
    matnm_add(lm->mat, lm->mat, lm->mat_diff, ctx->in_size, ctx->out_size);
    return 1;
}
