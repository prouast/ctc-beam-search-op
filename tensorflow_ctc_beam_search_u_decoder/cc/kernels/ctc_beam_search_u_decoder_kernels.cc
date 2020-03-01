#define EIGEN_USE_THREADS

#include <limits>

#include "../util/ctc_beam_search_u_decoder.h"

//#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
//#include "tensorflow/core/framework/register_types.h"
//#include "tensorflow/core/framework/types.h"
//#include "tensorflow/core/lib/core/status.h"
//#include "tensorflow/core/platform/logging.h"
//#include "tensorflow/core/platform/macros.h"
//#include "tensorflow/core/util/sparse/sparse_tensor.h"
//#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

template <typename T>
class CTCBeamSearchUDecoderOp : public OpKernel {
  public:
    explicit CTCBeamSearchUDecoderOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("merge_repeated", &merge_repeated_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("beam_width", &beam_width_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("blank_index", &blank_index_));
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
      auto inputs_t = inputs->tensor<T, 3>();
      auto seq_len_t = seq_len->vec<int32>();
      auto outputs_c_t = outputs_c->flat<int32>();
      auto outputs_u_t = outputs_u->flat<int32>();
      //auto log_prob_t = log_prob->matrix<T>();
      //log_prob_t.setZero();

      // Save shape of inputs and specific input dimensions
      const TensorShape& inputs_shape = inputs->shape();
      const int64 max_time = inputs_shape.dim_size(0);
      const int64 batch_size = inputs_shape.dim_size(1);
      const int64 num_classes = inputs_shape.dim_size(2);

      // For every time step, copy elements into input_list_t
      std::vector<typename TTypes<T>::UnalignedConstMatrix> input_list_t;
      for (std::size_t t = 0; t < max_time; ++t) {
        input_list_t.emplace_back(inputs_t.data() + t * batch_size * num_classes,
                                  batch_size, num_classes);
      }

      ctc::CTCBeamSearchUDecoder<T> decoder(num_classes, blank_index_,
                                            beam_width_, &beam_scorer_, 1,
                                            merge_repeated_);

      Tensor input_chip(DataTypeToEnum<T>::v(), TensorShape({num_classes}));
      auto input_chip_t = input_chip.flat<T>();

      std::vector<std::vector<std::vector<int> > > best_paths(batch_size);
      std::vector<T> log_probs;

      // Iterate over all batch elements
      for (int b = 0; b < batch_size; ++b) {
        auto& best_paths_b = best_paths[b];
        best_paths_b.resize(GetTopPaths());
        // Iterate over all time steps
        for (int t = 0; t < seq_len_t(b); ++t) {
          input_chip_t = input_list_t[t].chip(b, 0);
          auto input_bi = Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>(
            input_chip_t.data(), num_classes);
          std::cout << "------------ b=" << b << " t=" << t << " ------------" << std::endl;
          std::cout << input_bi << std::endl;
          // beam search step
          // TODO understand why test result is not as expected
          decoder.Step(input_bi);
        }
        // Get top paths
        // Get top uncollapsed paths
        // beam search Reset
      }
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

    inline int GetTopPaths() const { return top_paths_; }
    void SetTopPaths(int tp) { top_paths_ = tp; }

  private:
    typename ctc::CTCBeamSearchUDecoder<T>::DefaultBeamScorer beam_scorer_;
    bool merge_repeated_;
    int beam_width_;
    int blank_index_;
    int def_val_;
    int top_paths_;
    TF_DISALLOW_COPY_AND_ASSIGN(CTCBeamSearchUDecoderOp);
};

#define REGISTER_CPU(T)                                                       \
REGISTER_KERNEL_BUILDER(                                                      \
      Name("CTCBeamSearchUDecoder").Device(DEVICE_CPU).TypeConstraint<T>("T"),  \
      CTCBeamSearchUDecoderOp<T>);

REGISTER_CPU(float);
REGISTER_CPU(double);

#undef REGISTER_CPU

}  // end namespace tensorflow
