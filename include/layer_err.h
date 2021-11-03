#ifndef LAYER_ERR_H
#define LAYER_ERR_H

#include "layer.h"

struct layer_err{
    struct layer interface;
    float *ref;
};

struct layer_err *layer_err_init(struct layer_err *dst, struct node_ctx in_ctx, struct node_ctx out_ctx);
void layer_err_free(struct layer *dst);

int layer_err_forward(struct layer *ctx);
int layer_err_backward(struct layer *ctx);
int layer_err_update(struct layer *ctx, float s);

static struct layer_ops layer_err_ops = {
    .free = layer_err_free,
    .forward = layer_err_forward,
    .backward = layer_err_backward,
    .update = layer_err_update
};

#endif //LAYER_ERR_H
