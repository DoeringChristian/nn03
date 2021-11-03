#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"
#include "layer_init.h"
#include "node.h"
#include "darray.h"

struct network{
    darray(struct layer *) layers;
    darray(struct node *) nodes;
    size_t depth;
};

//struct network *network_init(struct network *dst, const size_t *sizes, const enum layer_type *types, size_t depth);
struct network *nework_init(struct network *dst);
void network_free(struct network *dst);

int network_node_push_back(struct network *dst, size_t node_size);
int network_layer_push_back(struct network *dst, struct layer *src);

int network_forward(struct network *dst);
int network_backward(struct network *dst);
int network_update(struct network *dst, float s);

#endif //NETWORK_H
