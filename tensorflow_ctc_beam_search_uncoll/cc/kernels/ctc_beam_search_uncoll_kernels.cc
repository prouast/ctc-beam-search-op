//#define EIGEN_USE_THREADS

//#include <limits>

//#include "tensorflow/core/util/ctc/ctc_beam_search.h"
//#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
//#include "tensorflow/core/framework/types.h"
//#include "tensorflow/core/lib/core/status.h"
//#include "tensorflow/core/platform/logging.h"
//#include "tensorflow/core/platform/macros.h"
//#include "tensorflow/core/util/sparse/sparse_tensor.h"

using namespace tensorflow;

class CTCBeamSearchUncolldOp : public OpKernel {
  public:
    explicit CTCBeamSearchUncollOp(OpKernelContext *ctx) : OpKernel(ctx) {
      //OP_REQUIRES_OK(ctx, ctx->GetAttr("beam_width", &beam_width_));
      //OP_REQUIRES_OK(ctx, ctx->GetAttr("blank", &blank));
      //OP_REQUIRES_OK(ctx, ctx->GetAttr("def_val", &def_val));
    }

    void Compute(OpKernelContext* ctx) override {
      // Grab the input tensor
      const Tensor& input_tensor = context->input(0);
      auto input = input_tensor.flat<int32>();

      // Create an output tensor
      Tensor* output_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                       &output_tensor));
      auto output_flat = output_tensor->flat<int32>();

      // Set all but the first element of the output tensor to 0.
      const int N = input.size();
      for (int i = 1; i < N; i++) {
        output_flat(i) = 0;
      }

      // Preserve the first input value if possible.
      if (N > 0) output_flat(0) = input(0);

      // Read inputs and allocate outputs
      //const Tensor* inputs;
      //const Tensor* seq_len;
      //Tensor* log_prob = nullptr;
      //OpOutputList decoded_indices;
      //OpOutputList decoded_values;
      //OpOutputList decoded_shape;
      //OP_REQUIRES_OK(ctx, ValidateInputsGenerateOutputs(ctx, &inputs, &seq_len,
      //  &log_prob, &decoded_indices, &decoded_values, &decoded_shape));
      // Save variables as specific tensorflow types
      //auto inputs_t = inputs->tensor<float, 3>();
      //auto seq_len_t = seq_len->vec<int32>();
      //auto log_prob_t = log_prob->matrix<float>();
      //log_prob_t.setZero();
      // Save shape of inputs and specific input dimensions
      //const TensorShape& inputs_shape = inputs->shape();
      //const int64 max_time = inputs_shape.dim_size(0);
      //const int64 batch_size = inputs_shape.dim_size(1);
      //const int64 num_classes = inputs_shape.dim_size(2);
      // For every time step, copy  ???
      //std::vector<TTypes<float>::UnalignedConstMatrix> input_list_t;
      //for (std::size_t t = 0; t < max_time; ++t) {
      //  input_list_t.emplace_back(inputs_t.data() + t * batch_size * num_classes, batch_size, num_classes);
      //}
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
    }

    //Status ValidateInputsGenerateOutputs(OpKernelContext *ctx,
    //  const Tensor** inputs, const Tensor** seq_len, Tensor** log_prob,
    //  OpOutputList* decoded_indices, OpOutputList* decoded_values,
    //  OpOutputList* decoded_shape) const {
    //  // Fetch inputs from context
    //  Status status = ctx->input("inputs", inputs);
    //  if (!status.ok()) return status;
    //  // Fetch sequence length from context
    //  status = ctx->input("sequence_length", seq_len);
    //  if (!status.ok()) return status;
    //  // Fetch shape of inputs
    //  const TensorShape& inputs_shape = (*inputs)->shape();
    //  // Throw error if input does not have 3 dims
    //  if (inputs_shape.dims() != 3) {
    //    return errors::InvalidArgument("inputs is not a 3-Tensor");
    //  }
    //  // Fetch sizes of individual dimensions
    //  const int64 max_time = inputs_shape.dim_size(0);
    //  const int64 batch_size = inputs_shape.dim_size(1);
    //  // Throw error if max time is 0
    //  if (max_time == 0) {
    //    return errors::InvalidArgument("max_time is 0");
    //  }
    //  // Throw error if sequence length is not a vector
    //  if (!TensorShapeUtils::IsVector((*seq_len)->shape())) {
    //    return errors::InvalidArgument("sequence_length is not a vector");
    //  }
    //  // Throw error if dim of sequence length is not the same as batch size
    //  if (!(batch_size == (*seq_len)->dim_size(0))) {
    //    return errors::FailedPrecondition(
    //        "len(sequence_length) != batch_size.  ", "len(sequence_length):  ",
    //        (*seq_len)->dim_size(0), " batch_size: ", batch_size);
    //  }
    //  // sequence length as int32 vector
    //  auto seq_len_t = (*seq_len)->vec<int32>();
    //  // Throw error if sequence length is not always less than max time
    //  for (int b = 0; b < batch_size; ++b) {
    //    if (!(seq_len_t(b) <= max_time)) {
    //      return errors::FailedPrecondition("sequence_length(", b, ") <= ",
    //                                        max_time);
    //    }
    //  }
    //  // Allocate log probability output
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
    //  // Return OK
    //  return Status::OK();
    //}

  //private:
  //  bool merge_repeated_;
  //  int beam_width_;
  //  TF_DISALLOW_COPY_AND_ASSIGN(CTCBeamSearchDecoderOp);
};

REGISTER_KERNEL_BUILDER(Name("CTCBeamSearchUncoll").Device(DEVICE_CPU),
  CTCBeamSearchUncollOp);
