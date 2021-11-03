#include "network.h"

struct network *nework_init(struct network *dst){
    darray_init(&dst->layers, 16);
    darray_init(&dst->nodes, 16);

    return dst;
}
void network_free(struct network *dst){
    for(size_t i = 0;i < darray_len(&dst->nodes);i++){
        node_free(dst->nodes[i]);
        free(dst->nodes[i]);
    }
    darray_free(&dst->layers);
    darray_free(&dst->nodes);
}

int network_node_push_back(struct network *dst, size_t node_size){
    struct node *node = malloc(sizeof(struct node));
    node_init(node, node_size);
    darray_push_back(&dst->nodes, node);
    return 1;
}
int network_layer_push_back(struct network *dst, struct layer *src){
    darray_push_back(&dst->layers, src);

    // maybe not correct use of darray
    node_ctx_set_node(&src->in_ctx, &dst->nodes[darray_len(&dst->layers) -1]);

    return 1;
}
int network_forward(struct network *dst){
    for(size_t i = 0;i < darray_len(&dst->layers);i++)
        layer_forward(dst->layers[i]);
    return 1;
}
int network_backward(struct network *dst){
    for(size_t i = darray_len(&dst->layers) - 1;i >= 0; i++)
        layer_backward(dst->layers[i]);
    return 1;
}
int network_update(struct network *dst, float s){
    for(size_t i = 0; i < darray_len(&dst->layers);i++)
        layer_update(dst->layers[i], s);
    return 1;
}
