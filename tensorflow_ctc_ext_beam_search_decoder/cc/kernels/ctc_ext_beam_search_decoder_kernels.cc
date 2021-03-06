#define EIGEN_USE_THREADS

#include <limits>
#include "../util/ctc_ext_beam_search_decoder.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

template <typename T>
class CTCExtBeamSearchDecoderOp : public OpKernel {
  public:
    explicit CTCExtBeamSearchDecoderOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("merge_repeated", &merge_repeated_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("beam_width", &beam_width_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("blank_index", &blank_index_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("blank_label", &blank_label_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("top_paths", &top_paths_));
    }

    void Compute(OpKernelContext* ctx) override {
      // Read inputs and allocate outputs
      const Tensor* inputs;
      const Tensor* seq_len;
      OpOutputList dec_indices;
      OpOutputList dec_values;
      OpOutputList dec_shape;
      OpOutputList ali_indices;
      OpOutputList ali_values;
      OpOutputList ali_shape;
      Tensor* log_prob = nullptr;
      OP_REQUIRES_OK(ctx, ValidateInputsGenerateOutputs(ctx, &inputs, &seq_len,
        &log_prob, &dec_indices, &dec_values, &dec_shape,
        &ali_indices, &ali_values, &ali_shape));

      // Save variables as specific types
      auto inputs_t = inputs->tensor<T, 3>();
      auto seq_len_t = seq_len->vec<int32>();
      auto log_prob_t = log_prob->matrix<T>();
      log_prob_t.setZero();

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

      // The decoder
      ctc::CTCExtBeamSearchDecoder<T> decoder(num_classes, blank_index_,
                                            beam_width_, &beam_scorer_,
                                            blank_label_, 1, merge_repeated_);

      Tensor input_chip(DataTypeToEnum<T>::v(), TensorShape({num_classes}));
      auto input_chip_t = input_chip.flat<T>();

      // Store results
      std::vector<std::vector<std::vector<int>>> best_paths(batch_size);
      std::vector<std::vector<std::vector<int>>> best_alignments(batch_size);
      std::vector<T> log_probs;

      // Iterate over all batch elements
      for (int b = 0; b < batch_size; ++b) {
        auto& best_paths_b = best_paths[b];
        auto& best_alignments_b = best_alignments[b];
        best_paths_b.resize(top_paths_);
        best_alignments_b.resize(top_paths_);
        // Iterate over all time steps
        for (int t = 0; t < seq_len_t(b); ++t) {
          input_chip_t = input_list_t[t].chip(b, 0);
          auto input_bi = Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>(
            input_chip_t.data(), num_classes);
          decoder.Step(input_bi);
        }
        // Get top paths
        OP_REQUIRES_OK(
          ctx, decoder.TopPaths(top_paths_, &best_paths_b, &best_alignments_b,
                                &log_probs, merge_repeated_));
        // beam search Reset
        decoder.Reset();
        // Copy log probs
        for (int bp = 0; bp < top_paths_; ++bp) {
          log_prob_t(b, bp) = log_probs[bp];
        }
      }
      // Store all decoded sequences
      OP_REQUIRES_OK(ctx, StoreAllDecodedSequences(
        best_paths, best_alignments, &dec_indices, &dec_values, &dec_shape,
        &ali_indices, &ali_values, &ali_shape));
    }

    Status ValidateInputsGenerateOutputs(OpKernelContext *ctx,
      const Tensor** inputs, const Tensor** seq_len, Tensor** log_prob,
      OpOutputList* dec_indices, OpOutputList* dec_values,
      OpOutputList* dec_shape, OpOutputList* ali_indices,
      OpOutputList* ali_values, OpOutputList* ali_shape) const {
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
      // Allocate list of outputs for alignments
      s = ctx->output_list("alignment_indices", ali_indices);
      if (!s.ok()) return s;
      s = ctx->output_list("alignment_values", ali_values);
      if (!s.ok()) return s;
      s = ctx->output_list("alignment_shape", ali_shape);
      if (!s.ok()) return s;
      // Return OK
      return Status::OK();
    }

    // sequences[b][p][ix] stores decoded value "ix" of path "p" for batch "b".
    Status StoreAllDecodedSequences(
      const std::vector<std::vector<std::vector<int> > >& sequences,
      const std::vector<std::vector<std::vector<int> > >& alignments,
      OpOutputList* dec_indices, OpOutputList* dec_values,
      OpOutputList* dec_shape, OpOutputList* ali_indices,
      OpOutputList* ali_values, OpOutputList* ali_shape) const {

      // Calculate the total number of entries for each path
      const int64 batch_size = sequences.size();
      std::vector<int64> num_entries_dec(top_paths_, 0);
      std::vector<int64> num_entries_ali(top_paths_, 0);

      // Calculate num_entries per path
      for (const auto& batch_s : sequences) {
        CHECK_EQ(batch_s.size(), top_paths_);
        for (int p = 0; p < top_paths_; ++p) {
          num_entries_dec[p] += batch_s[p].size();
        }
      }

      // Calculate num_entries per alignment
      for (const auto& batch_s : alignments) {
        CHECK_EQ(batch_s.size(), top_paths_);
        for (int p = 0; p < top_paths_; ++p) {
          num_entries_ali[p] += batch_s[p].size();
        }
      }

      for (int p = 0; p < top_paths_; ++p) {
        Tensor* p_dec_indices = nullptr;
        Tensor* p_dec_values = nullptr;
        Tensor* p_dec_shape = nullptr;
        Tensor* p_ali_indices = nullptr;
        Tensor* p_ali_values = nullptr;
        Tensor* p_ali_shape = nullptr;

        const int64 p_num_dec = num_entries_dec[p];
        const int64 p_num_ali = num_entries_ali[p];

        Status s = dec_indices->allocate(p, TensorShape({p_num_dec, 2}), &p_dec_indices);
        if (!s.ok()) return s;
        s = dec_values->allocate(p, TensorShape({p_num_dec}), &p_dec_values);
        if (!s.ok()) return s;
        s = dec_shape->allocate(p, TensorShape({2}), &p_dec_shape);
        if (!s.ok()) return s;
        s = ali_indices->allocate(p, TensorShape({p_num_ali, 2}), &p_ali_indices);
        if (!s.ok()) return s;
        s = ali_values->allocate(p, TensorShape({p_num_ali}), &p_ali_values);
        if (!s.ok()) return s;
        s = ali_shape->allocate(p, TensorShape({2}), &p_ali_shape);
        if (!s.ok()) return s;

        auto dec_indices_t = p_dec_indices->matrix<int64>();
        auto dec_values_t = p_dec_values->vec<int64>();
        auto dec_shape_t = p_dec_shape->vec<int64>();
        auto ali_indices_t = p_ali_indices->matrix<int64>();
        auto ali_values_t = p_ali_values->vec<int64>();
        auto ali_shape_t = p_ali_shape->vec<int64>();

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

        max_decoded = 0;
        offset = 0;

        for (int64 b = 0; b < batch_size; ++b) {
          auto& p_batch = alignments[b][p];
          int64 num_decoded = p_batch.size();
          max_decoded = std::max(max_decoded, num_decoded);
          std::copy_n(p_batch.begin(), num_decoded, &ali_values_t(offset));
          for (int64 t = 0; t < num_decoded; ++t, ++offset) {
            ali_indices_t(offset, 0) = b;
            ali_indices_t(offset, 1) = t;
          }
        }

        ali_shape_t(0) = batch_size;
        ali_shape_t(1) = max_decoded;
      }
      return Status::OK();
    }

  private:
    typename ctc::CTCExtBeamSearchDecoder<T>::DefaultBeamScorer beam_scorer_;
    bool merge_repeated_;
    int beam_width_;
    int blank_index_;
    int blank_label_;
    int top_paths_;
    TF_DISALLOW_COPY_AND_ASSIGN(CTCExtBeamSearchDecoderOp);
};

#define REGISTER_CPU(T)                                                       \
REGISTER_KERNEL_BUILDER(                                                      \
      Name("CTCExtBeamSearchDecoder").Device(DEVICE_CPU).TypeConstraint<T>("T"),  \
      CTCExtBeamSearchDecoderOp<T>);

REGISTER_CPU(float);
REGISTER_CPU(double);

#undef REGISTER_CPU

}  // end namespace tensorflow
