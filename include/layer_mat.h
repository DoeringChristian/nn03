#ifndef LAYER_MAT_H
#define LAYER_MAT_H

#include "layer.h"

struct layer_mat{
    struct layer interface;
    float *mat, *mat_diff;
};

struct layer_mat *layer_mat_init(struct layer_mat *dst, struct node_ctx in_ctx, struct node_ctx out_ctx);
void layer_mat_free(struct layer *dst);

int layer_mat_forward(struct layer *ctx);
int layer_mat_backward(struct layer *ctx);
int layer_mat_update(struct layer *ctx, float s);

static struct layer_ops layer_mat_ops = {
    .free = layer_mat_free,
    .forward = layer_mat_forward,
    .backward = layer_mat_backward,
    .update = layer_mat_update,
};

#endif //LAYER_MAT_H
