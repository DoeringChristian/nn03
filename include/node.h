#ifndef NODE_H
#define NODE_H

#include "nmath.h"

#ifndef MIN
#define MIN(_x, _y) ((_x) > (_y) ? (_y) : (_x))
#endif //MIN

struct node;

struct node_ctx{
    struct node *node;
    size_t n;
};
struct node_ctx node_ctx(struct node *node, size_t n);

struct node{
    float *state, *grad;
    size_t n;
};
struct node *node_init(struct node *dst, size_t n);
struct node node(size_t n);
void node_free(struct node *dst);

static inline float *node_ctx_state(struct node_ctx *src){
    return src->node->state;
}
static inline float *node_ctx_grad(struct node_ctx *src){
    return src->node->grad;
}
static inline int node_ctx_valid(struct node_ctx *src){
    return (src->node != NULL && src->node->state != NULL && src->node->grad != NULL);
}

static inline int node_ctx_set_node(struct node_ctx *dst, struct node *node){
    if(dst->node != NULL)
        return 0;
    dst->node = node;
    return 1;
}

#endif //NODE_H
