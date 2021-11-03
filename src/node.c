#include "node.h"

struct node_ctx node_ctx(struct node *node, size_t n){
    return (struct node_ctx){.node = node, .n = MIN(n, node->n)};
}

struct node *node_init(struct node *dst, size_t n){
    dst->n = n;
    dst->grad = vecn_new(n);
    dst->state = vecn_new(n);
    return dst;
}
struct node node(size_t n){
    return (struct node){.n = n, .state = vecn_new(n), .grad = vecn_new(n)};
}
void node_free(struct node *dst){
    vecn_free(dst->state);
    vecn_free(dst->grad);
    dst->state = NULL;
    dst->grad = NULL;
    dst->n = 0;
}
