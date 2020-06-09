#define EIGEN_USE_THREADS

#include <limits>
#include "../util/ctc_beam_search_decoder.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

class CTCDecodeHelper {
 public:
  CTCDecodeHelper() : top_paths_(1) {}

  inline int GetTopPaths() const { return top_paths_; }
  void SetTopPaths(int tp) { top_paths_ = tp; }

  Status ValidateInputsGenerateOutputs(OpKernelContext *ctx,
    const Tensor** inputs, const Tensor** seq_len, Tensor** log_prob,
    OpOutputList* dec_indices, OpOutputList* dec_values,
    OpOutputList* dec_shape) const {
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
    // Allocate log probability output
    Status s = ctx->allocate_output("log_probability",
      TensorShape({batch_size, top_paths_}), log_prob);
    if (!s.ok()) return s;
    // Allocate list of outputs for decoded
    s = ctx->output_list("decoded_indices", dec_indices);
    if (!s.ok()) return s;
    s = ctx->output_list("decoded_values", dec_values);
    if (!s.ok()) return s;
    s = ctx->output_list("decoded_shape", dec_shape);
    if (!s.ok()) return s;
    // Return OK
    return Status::OK();
  }

  // sequences[b][p][ix] stores decoded value "ix" of path "p" for batch "b".
  Status StoreAllDecodedSequences(
    const std::vector<std::vector<std::vector<int> > >& sequences,
    OpOutputList* dec_indices, OpOutputList* dec_values,
    OpOutputList* dec_shape) const {

    // Calculate the total number of entries for each path
    const int64 batch_size = sequences.size();
    std::vector<int64> num_entries_dec(top_paths_, 0);

    // Calculate num_entries per path
    for (const auto& batch_s : sequences) {
      CHECK_EQ(batch_s.size(), top_paths_);
      for (int p = 0; p < top_paths_; ++p) {
        num_entries_dec[p] += batch_s[p].size();
      }
    }

    for (int p = 0; p < top_paths_; ++p) {
      Tensor* p_dec_indices = nullptr;
      Tensor* p_dec_values = nullptr;
      Tensor* p_dec_shape = nullptr;

      const int64 p_num_dec = num_entries_dec[p];

      Status s = dec_indices->allocate(p, TensorShape({p_num_dec, 2}), &p_dec_indices);
      if (!s.ok()) return s;
      s = dec_values->allocate(p, TensorShape({p_num_dec}), &p_dec_values);
      if (!s.ok()) return s;
      s = dec_shape->allocate(p, TensorShape({2}), &p_dec_shape);
      if (!s.ok()) return s;

      auto dec_indices_t = p_dec_indices->matrix<int64>();
      auto dec_values_t = p_dec_values->vec<int64>();
      auto dec_shape_t = p_dec_shape->vec<int64>();

      int64 max_decoded = 0;
      int64 offset = 0;

      for (int64 b = 0; b < batch_size; ++b) {
        auto& p_batch = sequences[b][p];
        int64 num_decoded = p_batch.size();
        max_decoded = std::max(max_decoded, num_decoded);
        std::copy_n(p_batch.begin(), num_decoded, &dec_values_t(offset));
        for (int64 t = 0; t < num_decoded; ++t, ++offset) {
          dec_indices_t(offset, 0) = b;
          dec_indices_t(offset, 1) = t;
        }
      }

      dec_shape_t(0) = batch_size;
      dec_shape_t(1) = max_decoded;
    }
    return Status::OK();
  }

 private:
  int top_paths_;
  TF_DISALLOW_COPY_AND_ASSIGN(CTCDecodeHelper);
};

template <typename T>
class CTCBeamSearchDecoderOp : public OpKernel {
  public:
    explicit CTCBeamSearchDecoderOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("merge_repeated", &merge_repeated_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("beam_width", &beam_width_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("blank_index", &blank_index_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("blank_label", &blank_label_));
      int top_paths;
      OP_REQUIRES_OK(ctx, ctx->GetAttr("top_paths", &top_paths));
      decode_helper_.SetTopPaths(top_paths);
    }

    void Compute(OpKernelContext* ctx) override {
      const Tensor* inputs;
      const Tensor* seq_len;
      Tensor* log_prob = nullptr;
      OpOutputList dec_indices;
      OpOutputList dec_values;
      OpOutputList dec_shape;
      OP_REQUIRES_OK(ctx, decode_helper_.ValidateInputsGenerateOutputs(
        ctx, &inputs, &seq_len, &log_prob, &dec_indices, &dec_values, &dec_shape));

      auto inputs_t = inputs->tensor<T, 3>();
      auto seq_len_t = seq_len->vec<int32>();
      auto log_prob_t = log_prob->matrix<T>();

      const TensorShape& inputs_shape = inputs->shape();

      const int64 max_time = inputs_shape.dim_size(0);
      const int64 batch_size = inputs_shape.dim_size(1);
      const int64 num_classes_raw = inputs_shape.dim_size(2);
      const int num_classes = static_cast<const int>(num_classes_raw);

      log_prob_t.setZero();

      std::vector<typename TTypes<T>::UnalignedConstMatrix> input_list_t;

      for (std::size_t t = 0; t < max_time; ++t) {
        input_list_t.emplace_back(inputs_t.data() + t * batch_size * num_classes,
                                  batch_size, num_classes);
      }

      // The decoder
      ctc::CTCBeamSearchDecoder<T> decoder(num_classes, blank_index_,
                                            beam_width_, &beam_scorer_,
                                            blank_label_, 1, merge_repeated_);

      Tensor input_chip(DataTypeToEnum<T>::v(), TensorShape({num_classes}));
      auto input_chip_t = input_chip.flat<T>();

      // Store results
      std::vector<std::vector<std::vector<int>>> best_paths(batch_size);
      std::vector<T> log_probs;

      // Iterate over all batch elements
      for (int b = 0; b < batch_size; ++b) {
        auto& best_paths_b = best_paths[b];
        best_paths_b.resize(decode_helper_.GetTopPaths());
        // Iterate over all time steps
        for (int t = 0; t < seq_len_t(b); ++t) {
          input_chip_t = input_list_t[t].chip(b, 0);
          auto input_bi = Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>(
            input_chip_t.data(), num_classes);
          decoder.Step(input_bi);
        }
        // Get top paths
        OP_REQUIRES_OK(
          ctx, decoder.TopPaths(decode_helper_.GetTopPaths(), &best_paths_b, &log_probs, merge_repeated_));
        // beam search Reset
        decoder.Reset();
        // Copy log probs
        for (int bp = 0; bp < decode_helper_.GetTopPaths(); ++bp) {
          log_prob_t(b, bp) = log_probs[bp];
        }
      }
      // Store all decoded sequences
      OP_REQUIRES_OK(ctx, decode_helper_.StoreAllDecodedSequences(
        best_paths, &dec_indices, &dec_values, &dec_shape));
    }

  private:
    CTCDecodeHelper decode_helper_;
    typename ctc::CTCBeamSearchDecoder<T>::DefaultBeamScorer beam_scorer_;
    bool merge_repeated_;
    int beam_width_;
    int blank_index_;
    int blank_label_;
    TF_DISALLOW_COPY_AND_ASSIGN(CTCBeamSearchDecoderOp<T>);
};

#define REGISTER_CPU(T)                                                       \
REGISTER_KERNEL_BUILDER(                                                      \
      Name("CTCBeamSearchDecoder_").Device(DEVICE_CPU).TypeConstraint<T>("T"),  \
      CTCBeamSearchDecoderOp<T>);

REGISTER_CPU(float);
REGISTER_CPU(double);

#undef REGISTER_CPU

}  // end namespace tensorflow
