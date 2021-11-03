#ifndef LAYER_H
#define LAYER_H

#include "nmath.h"
#include "node.h"

enum layer_type{
    LAYER_TYPE_MAT,
    LAYER_TYPE_PARALLEL,
    LAYER_TYPE_ERR,
};

struct layer_ops;

// in and in_grad / out and out_grad could be implemented as nx2 matrix
struct layer{
    struct layer_ops *ops;

    struct node_ctx in_ctx, out_ctx;
};

struct layer_ops{
    void (*free)(struct layer *dst);
    int (*forward)(struct layer *ctx);
    int (*backward)(struct layer *ctx);
    int (*update)(struct layer *ctx, float s);
};

static inline void layer_free(struct layer *dst){
    if(dst->ops != NULL && dst->ops->free != NULL)
        dst->ops->free(dst);
}

static inline int layer_forward(struct layer *ctx){
    if(ctx->ops != NULL && ctx->ops->forward != NULL)
        return ctx->ops->forward(ctx);
    return 0;
}

static inline int layer_backward(struct layer *ctx){
    if(ctx->ops != NULL && ctx->ops->backward != NULL)
        return ctx->ops->backward(ctx);
    return 0;
}

static inline int layer_update(struct layer *ctx, float s){
    if(ctx->ops != NULL && ctx->ops->update != NULL)
        return ctx->ops->update(ctx, s);
    return 0;
}

#endif //LAYER_H
