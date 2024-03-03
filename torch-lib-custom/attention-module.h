class KQVModule;

class AttentionModule {
    public:
        /** Children Modules */

        // Key fed into this module
        KQVModule* key_;
        // Query fed into this module
        KQVModule* query_;
        // Value fed into this module
        KQVModule* value_;

        /** Empty init constructor to avoid lazy init errors for now. */
        // AttentionModule() {}

        AttentionModule(int context_size, 
                        int head_size,
                        bool is_decoder, 
                        KQVModule &key,
                        KQVModule &query,
                        KQVModule &value) {
            context_size_ = context_size;
            head_size_ = head_size;
            is_decoder_ = is_decoder;
            options_ = torch::TensorOptions()
                .dtype(torch::kFloat32)
                .layout(torch::kStrided)
                .requires_grad(true);
            // Generate a square tensor with given dimensions using the key and query.
            // tensor_ = torch::rand({context_size_, context_size_}, options_);

            // assign the children modules below
            key_ = &key;
            query_ = &query;
            value_ = &value;
        }

        /**
         * Get the tensor for this attention module;
         */
        torch::Tensor getTensor() {
            return tensor_;
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

        /** 
         * Feeds the Value transformation result through attention matrix. 
         *  
         *  @return the resulting matrix.
         */
        static torch::Tensor forward(AttentionModule *attention,
                torch::Tensor *input);

    private:
        /** 
         * Initializes the (T, T) attention matrix.
         *
         * @param input - the input fed into key and query transformations.
         */
        static void initTensor(AttentionModule *attention, torch::Tensor *input);

        /** Internal representation and hyper parameters */

        // Sequence size or number of tokens we consider.
        int context_size_;
        // Single Head size
        int head_size_;
        // are we masking the matrix or not
        bool is_decoder_;
        // The current set of options for internal tensor.
        torch::TensorOptions options_;
        // Internal matrix representation for self-attention.
        torch::Tensor tensor_;
};

