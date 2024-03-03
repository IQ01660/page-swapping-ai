/**
 * This file contains the key, query, and value blocks,
 * that are used as a part of each attention head in
 * parallel (usually).
 */

// class declarations
#include <cstdio>
#include <cstring>
class AttentionModule;

class KQVModule {
    public:
        int channel_count_;
        int head_size_;

        KQVModule(
                int channel_count, 
                int head_size, 
                const char* non_linearity="none") {

            channel_count_ = channel_count;
            head_size_ = head_size;
            non_linearity_ = non_linearity;
            options_ = torch::TensorOptions()
                .dtype(torch::kFloat32)
                .layout(torch::kStrided)
                .requires_grad(true);
            tensor_ = torch::rand({channel_count_, head_size_}, options_);
            tensor_.retain_grad();
        }

        torch::Tensor getTensor() {
            return tensor_;
        }

        /** 
         * Feeds the input into the linear transformation. 
         *  
         * @return the resulting matrix.
         */
        torch::Tensor forward(torch::Tensor *input) {
            torch::Tensor toReturn = at::matmul(*input, tensor_);
            toReturn.requires_grad_(true);
            toReturn.retain_grad();
            if (strcmp(non_linearity_, "tanh") == 0) {
                toReturn = at::tanh(toReturn);
                toReturn.requires_grad_(true);
                toReturn.retain_grad();
            }
            return toReturn;
        }

        void clear_grad() {
            tensor_.detach();
            tensor_.requires_grad_(true);
            tensor_.retain_grad();
        }

        /**
         * The actual part where the model learns the parameters.
         *
         * Done by calculating the Jacobian matrix after back propagation
         * has already been carried out. The model, then, uses the gradient
         * vectors to update the parameters in order to minimize the loss function.
         */
        void update_params(int eta) {
            torch::Tensor jacobian = tensor_.grad();
            tensor_ = tensor_ + (-1 * eta * jacobian);
        }

    private:
        // The current set of options for internal tensor.
        torch::TensorOptions options_;
        // Internal matrix representation for this transformation.
        torch::Tensor tensor_;

        // non-linearity type after #forward()
        const char* non_linearity_;
};
