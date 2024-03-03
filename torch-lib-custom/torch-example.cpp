/**
 * This file contains any helper functions that are
 * callable from C code that help create and learn
 * a neural net using PyTorch C++ API.
 */

#include <cmath>
#include <cstdio>
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include "torch-interface.h"
#include "linear-layer.h"
#include "single-head.h"

torch::Tensor forward(torch::Tensor&, SingleHead*, KQVModule*, KQVModule*);
void update_params(SingleHead*, KQVModule*, KQVModule*);
void clear_grad(SingleHead*, KQVModule*, KQVModule*);


namespace F = torch::nn::functional;

/**
 * Hyperparameters
 */
namespace hyperparams {
    // output 
    int pages = 100;
    int label_count = 3;
    // input 
    int batch_size = 10;
    int channel_size = 8;
    int context_size = 10;
    // inner architecture
    int head_size = 4;
    int ffn_inner_layer_size = 100;
    int ffn_external_layer_size = pages;
    const char* external_non_linearity = "linear";
    // learning
    float eta = 1.0f; // TODO: issue with < 1 eta (floats?)
    int epochs = 1000;
}

/**
 * Various helpers for debugging
 */

namespace debugging {
    void printMatrix(torch::Tensor tensor) {
        std::cout << tensor << std::endl;
    }
}

/**
 * AttentionModule static method definitions.
 */

void AttentionModule::initTensor(
        AttentionModule *attention, 
        torch::Tensor *input) {
    // this should yield a square matrix of size (seq_length, seq_length)
    attention->tensor_ = at::matmul(attention->key_->forward(input), 
            attention->query_->forward(input).transpose(-2, -1));

    // scale the attention matrix to reach unit variance
    attention->tensor_ = 
        attention->tensor_ * (powf((float) attention->head_size_, -0.5f));

    if (attention->is_decoder_) {
        // create the mask for the square tensor product
        torch::Tensor upper_tri_tensor = torch::ones(
                {hyperparams::context_size, hyperparams::context_size}, at::kBool);
        upper_tri_tensor = at::triu(upper_tri_tensor, 1 /* diagonal is off */);
        
        // apply the mask with -INFINITY for passing cells
        attention->tensor_ = at::masked_fill(attention->tensor_, upper_tri_tensor, 
                at::Scalar(-INFINITY));
    }

    // softmax all row vectors
    attention->tensor_ = at::softmax(attention->tensor_, -1 /* dimension */);
}

torch::Tensor AttentionModule::forward(
        AttentionModule *attention, 
        torch::Tensor *input) {
    initTensor(attention, input);

    torch::Tensor toReturn = at::matmul(
            attention->getTensor(), 
            attention->value_->forward(input));

    return toReturn;
}

/**
 * This is declared as an extern C function
 * for tests in C swapper code.
 */
float* get_page() {
    // TODO: remove the dummy input and label
    //
    // dummy input
    torch::Tensor input = torch::rand({hyperparams::batch_size,
            hyperparams::context_size, 
            hyperparams::channel_size});

    // dummy target
    torch::Tensor target = torch::rand({hyperparams::batch_size, 
            hyperparams::context_size, 
            hyperparams::pages});
    target = at::softmax(target, -1 /* dimension */); 

    // ========================================
    // ========================================

    // single head of attention
    SingleHead head(
            hyperparams::context_size, 
            hyperparams::channel_size,
            hyperparams::head_size, NULL /* ignore - lazy-init */);

    KQVModule ffn_inner_layer   (hyperparams::head_size, 
            hyperparams::ffn_inner_layer_size);
    KQVModule ffn_external_layer(hyperparams::ffn_inner_layer_size, 
            hyperparams::ffn_external_layer_size, 
            hyperparams::external_non_linearity);

    // TODO: Add bias to linear layers
    // TODO: Create a training module that takes in training data from stdin
    // TODO: Add multiple heads if needed


    for (int i = 0; i < hyperparams::epochs; i++) {
        // forward propogation returns the last raw tensor
        torch::Tensor external_layer_out = forward(input, 
                &head, &ffn_inner_layer, &ffn_external_layer);

        // calculate the loss using cross-entropy
        auto loss = F::cross_entropy(external_layer_out, target);

        // print the loss
        std::cout << loss << std::endl;
        
        // clear jacobians (effectively grad().zero_())
        clear_grad(&head, &ffn_inner_layer, &ffn_external_layer);

        // run back-propogation
        loss.backward();

        // update all params using the jacobians
        update_params(&head, &ffn_inner_layer, &ffn_external_layer);
    }

    return 0;
}

torch::Tensor forward(
        torch::Tensor &input,
        SingleHead* head, 
        KQVModule* ffn_inner_layer, 
        KQVModule* ffn_external_layer) {
    auto single_head_result = head->forward(&input);
    auto inner_layer_out = ffn_inner_layer->forward(&single_head_result);
    torch::Tensor external_layer_out = ffn_external_layer->forward(&inner_layer_out);
    return external_layer_out;
}

void update_params(
        SingleHead* head, 
        KQVModule* ffn_inner_layer, 
        KQVModule* ffn_external_layer) {
    head->update_params(hyperparams::eta);
    ffn_inner_layer->update_params(hyperparams::eta);
    ffn_external_layer->update_params(hyperparams::eta);
}

void clear_grad(SingleHead *head,
        KQVModule* ffn_inner_layer, 
        KQVModule* ffn_external_layer) {
    head->attention_->key_->clear_grad();
    head->attention_->query_->clear_grad();
    head->attention_->value_->clear_grad();

    ffn_inner_layer->clear_grad();
    ffn_external_layer->clear_grad();
}
