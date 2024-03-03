/**
 * A wrapper to help modularize 
 * a single dot-product self-attention head.
 */

#include <cstdio>
#include <stdlib.h>
#include <torch/torch.h>
#include <iostream>
#include "attention-module.h"
#include "key-module.h"

// class declarations
class KQVModule;
class AttentionModule;

class SingleHead {
    public:
        AttentionModule *attention_;

        SingleHead(int context_size, int channel_count, int head_size, 
                AttentionModule *attention_lazy) {
            context_size_ = context_size;
            channel_count_ = channel_count;
            head_size_ = head_size;
            attention_ = attention_lazy;

            KQVModule *kqv = (KQVModule*) malloc(sizeof(KQVModule) * 3);
            if (kqv == NULL) {
                printf("couldn't malloc\n");
            } else {
                printf("malloc ptr=%p\n", kqv);
            }
            new (kqv)     KQVModule(channel_count, head_size);
            new (kqv + 1) KQVModule(channel_count, head_size);
            new (kqv + 2) KQVModule(channel_count, head_size);

            attention_ = (AttentionModule*) malloc(sizeof(AttentionModule));
            new (attention_) AttentionModule(context_size, head_size, 
                    true /* is_decoder */, 
                    kqv[0], kqv[1], kqv[2]);
        }

        /** 
         * Feeds the input into the single head. 
         *  
         * @return the resulting matrix.
         */
        torch::Tensor forward(torch::Tensor *input) {
            torch::Tensor toReturn = AttentionModule::forward(attention_, input);
            toReturn.requires_grad_(true);
            return toReturn;
        }

        /**
         * The actual part where the model learns the parameters.
         */
        void update_params(int eta) {
            // attention matrix itself is a product of K x Q so skip it

            // update the params for K, Q, V children
            attention_->key_->update_params(eta);
            attention_->query_->update_params(eta);
            attention_->value_->update_params(eta);
        }
    private:
        int context_size_;
        int channel_count_;
        int head_size_;
};
