//#define EIGEN_USE_THREADS

//#include <limits>
#include <cstdint>

//#include "tensorflow/core/util/ctc/ctc_beam_search.h"
//#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
//#include "tensorflow/core/framework/types.h"
//#include "tensorflow/core/lib/core/status.h"
//#include "tensorflow/core/platform/logging.h"
//#include "tensorflow/core/platform/macros.h"
//#include "tensorflow/core/util/sparse/sparse_tensor.h"

using namespace tensorflow;

class CTCBeamSearchUncollOp : public OpKernel {
  public:
    explicit CTCBeamSearchUncollOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("beam_width", &beam_width_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("blank", &blank_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("def_val", &def_val_));
    }

    void Compute(OpKernelContext* ctx) override {
      // Read inputs and allocate outputs
      const Tensor* inputs;
      const Tensor* seq_len;
      Tensor* outputs_c = NULL;
      Tensor* outputs_u = NULL;
      OP_REQUIRES_OK(ctx, ValidateInputsGenerateOutputs(ctx, &inputs, &seq_len,
        &outputs_c, &outputs_u));
      //Tensor* log_prob = nullptr;
      //OpOutputList decoded_indices;
      //OpOutputList decoded_values;
      //OpOutputList decoded_shape;
      //OP_REQUIRES_OK(ctx, ValidateInputsGenerateOutputs(ctx, &inputs, &seq_len,
      //  &log_prob, &decoded_indices, &decoded_values, &decoded_shape));

      // Save variables as specific types
      auto inputs_t = inputs->tensor<float, 3>();
      auto seq_len_t = seq_len->vec<int32>();
      auto outputs_c_t = outputs_c->flat<int32>();
      auto outputs_u_t = outputs_u->flat<int32>();
      //auto log_prob_t = log_prob->matrix<float>();
      //log_prob_t.setZero();

      // Save shape of inputs and specific input dimensions
      const TensorShape& inputs_shape = inputs->shape();
      const int64 max_time = inputs_shape.dim_size(0);
      const int64 batch_size = inputs_shape.dim_size(1);
      const int64 num_classes = inputs_shape.dim_size(2);

      // For every time step, copy elements into input_list_t
      std::vector<TTypes<float>::UnalignedConstMatrix> input_list_t;
      for (std::size_t t = 0; t < max_time; ++t) {
        input_list_t.emplace_back(inputs_t.data() + t * batch_size * num_classes,
                                  batch_size, num_classes);
      }

      
      // ...
      // Iterate over all batch elements
      //for (int b = 0; b < batch_size; ++b) {
      //  // ...
      //  // Iterate over all time steps
      //  for (int t = 0; t < seq_len_t(b); ++t) {
      //    // beam search step
      //  }
      //  // Get top paths
      //  // Get top uncollapsed paths
      //  // beam search Reset
      //}
      // Store all decoded sequences

      // Set all but the first element of the output tensor to 0.
      const int N = inputs_t.size();
      for (int i = 0; i < N; i++) {
        outputs_c_t(i) = 0;
        outputs_u_t(i) = 0;
      }

      // Set first output value to 1 / 2 if possible.
      if (N > 0) outputs_c_t(0) = beam_width_;
      if (N > 0) outputs_u_t(0) = beam_width_;

    }

    Status ValidateInputsGenerateOutputs(OpKernelContext *ctx,
      const Tensor** inputs, const Tensor** seq_len,
      Tensor** outputs_c, Tensor** outputs_u) const {
      //OpOutputList* decoded_indices, OpOutputList* decoded_values,
      //OpOutputList* decoded_shape) const {
      // Fetch inputs from context
      Status status = ctx->input("inputs", inputs);
      if (!status.ok()) return status;
      // Fetch sequence length from context
      status = ctx->input("sequence_length", seq_len);
      if (!status.ok()) return status;
      // Fetch shape of inputs
      const TensorShape& inputs_shape = (*inputs)->shape();
      // Throw error if input does not have 3 dims
      if (inputs_shape.dims() != 3) {
        return errors::InvalidArgument("inputs is not a 3-Tensor");
      }
      // Fetch sizes of individual dimensions
      const int64 max_time = inputs_shape.dim_size(0);
      const int64 batch_size = inputs_shape.dim_size(1);
      // Throw error if max time is 0
      if (max_time == 0) {
        return errors::InvalidArgument("max_time is 0");
      }
      // Throw error if sequence length is not a vector
      if (!TensorShapeUtils::IsVector((*seq_len)->shape())) {
        return errors::InvalidArgument("sequence_length is not a vector");
      }
      // Throw error if dim of sequence length is not the same as batch size
      if (!(batch_size == (*seq_len)->dim_size(0))) {
        return errors::FailedPrecondition(
            "len(sequence_length) != batch_size.  ", "len(sequence_length):  ",
            (*seq_len)->dim_size(0), " batch_size: ", batch_size);
      }
      // sequence length as int32 vector
      auto seq_len_t = (*seq_len)->vec<int32>();
      // Throw error if sequence length is not always less than max time
      for (int b = 0; b < batch_size; ++b) {
        if (!(seq_len_t(b) <= max_time)) {
          return errors::FailedPrecondition("sequence_length(", b, ") <= ",
                                            max_time);
        }
      }
      // Create output tensors
      Status s = ctx->allocate_output("decoded_c", inputs_shape, outputs_c);
      if (!s.ok()) return s;
      s = ctx->allocate_output("decoded_u", inputs_shape, outputs_u);
      if (!s.ok()) return s;
      // Allocate log probability output
    //  Status s = ctx->allocate_output(
    //      "log_probability", TensorShape({batch_size, top_paths_}), log_prob);
    //  if (!s.ok()) return s;
    //  // Allocate list of outputs for decoded
    //  s = ctx->output_list("decoded_indices", decoded_indices);
    //  if (!s.ok()) return s;
    //  s = ctx->output_list("decoded_values", decoded_values);
    //  if (!s.ok()) return s;
    //  s = ctx->output_list("decoded_shape", decoded_shape);
    //  if (!s.ok()) return s;
      // Return OK
      return Status::OK();
    }

  private:
    int beam_width_;
    int blank_;
    int def_val_;
    TF_DISALLOW_COPY_AND_ASSIGN(CTCBeamSearchUncollOp);
};

REGISTER_KERNEL_BUILDER(Name("CTCBeamSearchUncoll").Device(DEVICE_CPU),
  CTCBeamSearchUncollOp);
