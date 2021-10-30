#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"
#include "layer_init.h"

struct network{
    struct layer **layers;
    size_t *sizes;
    float **vecs;
    float **vecs_grad;
    size_t depth;
};

struct network *network_init(struct network *dst, const size_t *sizes, const enum layer_type *types, size_t depth);
void network_free(struct network *dst);

#endif //NETWORK_H
