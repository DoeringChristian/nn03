#ifndef LAYER_PARALLEL_H
#define LAYER_PARALLEL_H

#include "layer.h"
#include <math.h>

enum layer_parallel_type{
    LAYER_PARALLEL_TYPE_DEF,
    LAYER_PARALLEL_TYPE_SIGMOID,
};

struct layer_parallel{
    struct layer interface;

    float (*func)(float x);
    float (*func_diff)(float x);
};

struct layer_parallel *layer_parallel_init(struct layer_parallel *dst, float (*func)(float x), float (*func_diff)(float x), size_t size);
void layer_parallel_free(struct layer *dst);

int layer_parallel_forward(struct layer *ctx);
int layer_parallel_backward(struct layer *ctx);
int layer_parallel_update(struct layer *ctx, float s);

float layer_parallel_sigmoid(float x);
float layer_parallel_sigmoid_diff(float x);

static struct layer_ops layer_parallel_ops = {
    .free = layer_parallel_free,
    .forward = layer_parallel_forward,
    .backward = layer_parallel_backward,
    .update = layer_parallel_update,
};

#endif //LAYER_PARALLEL_H
