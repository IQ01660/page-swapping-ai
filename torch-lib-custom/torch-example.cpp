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

namespace F = torch::nn::functional;

/**
 * Hyperparameters
 */
namespace hyperparams {
    int pages = 100;
    int label_count = 3;
    int batch_size = 10;
    int channel_size = 8;
    int context_size = 10;
    int head_size = 4;
    int ffn_inner_layer_size = 100;
    int ffn_external_layer_size = pages;
    float eta = 0.01;
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

    // single head of attention
    SingleHead head(
            hyperparams::context_size, 
            hyperparams::channel_size,
            hyperparams::head_size, NULL /* ignore - lazy-init */);


    KQVModule ffn_inner_layer   (hyperparams::head_size, 
            hyperparams::ffn_inner_layer_size);
    KQVModule ffn_external_layer(hyperparams::ffn_inner_layer_size, 
            hyperparams::ffn_external_layer_size);

    // TODO: get a single forward() function
    torch::Tensor single_head_result = head.forward(&input);
    auto inner_layer_out = ffn_inner_layer.forward(&single_head_result);
    auto external_layer_out = ffn_external_layer.forward(&inner_layer_out);

    // TODO: Do normalization
    // TODO: Add bias to linear layers
    // TODO: Make the first linear layer ReLU
    // TODO: Make the second linear layer tanh one
    // TODO: Create a training module that takes in training data from stdin
    // TODO: Add multiple heads if needed

    auto loss = F::cross_entropy(external_layer_out, target);
    std::cout << "---------------" << std::endl;
    std::cout << loss << std::endl;
    std::cout << "---------------" << std::endl;

    loss.backward();

    debugging::printMatrix(head.attention_->getTensor());

    // TODO: get a single update_params() function
    head.update_params(hyperparams::eta);
    ffn_inner_layer.update_params(hyperparams::eta);
    ffn_external_layer.update_params(hyperparams::eta);

    debugging::printMatrix(head.attention_->getTensor());

    // TODO: get a single forward() function
    single_head_result = head.forward(&input);
    inner_layer_out = ffn_inner_layer.forward(&single_head_result);
    external_layer_out = ffn_external_layer.forward(&inner_layer_out);

    loss = F::cross_entropy(external_layer_out, target);
    std::cout << "---------------" << std::endl;
    std::cout << loss << std::endl;
    std::cout << "---------------" << std::endl;

    return 0;
}
