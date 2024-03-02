/**
 * Represents a single linear layer of neurons.
 */
class LinearLayer {
    public:
        // The internal tensor representing this layer.
        torch::Tensor tensor_;

        LinearLayer(int fan_in, int fan_out) {
            fan_in_ = fan_in;
            fan_out_ = fan_out;
            options_ = torch::TensorOptions()
                .dtype(torch::kFloat32)
                .layout(torch::kStrided)
                .requires_grad(true);
            // Generate a random tensor with given dimensions.
            // note: unit normal distribution is used by default
            tensor_ = torch::rand({fan_in_, fan_out_}, options_);
        }

        /**
         * Set custom options for the tensor.
         *
         * Will update the tensor (creating a new one) as a side effect.
         */
        void setOptions(torch::TensorOptions options) {
            options_ = options;
            tensor_ = tensor_.to(options_);
        }

    private:
        // The number of neurons taken in as inputs.
        int fan_in_;
        // The number of neurons in this layer.
        int fan_out_;
        // The current set of options for internal tensor.
        torch::TensorOptions options_;
};
