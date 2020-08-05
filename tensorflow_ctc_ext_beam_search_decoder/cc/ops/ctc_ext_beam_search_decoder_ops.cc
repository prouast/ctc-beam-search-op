#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;
using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("CTCExtBeamSearchDecoder")
    .Input("inputs: T")
    .Input("sequence_length: int32")
    .Attr("beam_width: int >= 1")
    .Attr("top_paths: int >= 1")
    .Attr("merge_repeated: bool = False")
    .Attr("blank_index: int = 0")
    .Attr("blank_label: int = -1")
    .Output("decoded_indices: top_paths * int64")
    .Output("decoded_values: top_paths * int64")
    .Output("decoded_shape: top_paths * int64")
    .Output("alignment_indices: top_paths * int64")
    .Output("alignment_values: top_paths * int64")
    .Output("alignment_shape: top_paths * int64")
    .Output("log_probability: T")
    .Attr("T: {float, double} = DT_FLOAT")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle inputs;
      ShapeHandle sequence_length;

      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &inputs));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &sequence_length));

      // Get batch size from inputs and sequence_length.
      DimensionHandle batch_size;
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(inputs, 1), c->Dim(sequence_length, 0), &batch_size));

      int32 top_paths;
      TF_RETURN_IF_ERROR(c->GetAttr("top_paths", &top_paths));

      // Outputs.
      int out_idx = 0;
      for (int i = 0; i < top_paths; ++i) {  // decoded_indices
        c->set_output(out_idx++, c->Matrix(InferenceContext::kUnknownDim, 2));
      }
      for (int i = 0; i < top_paths; ++i) {  // decoded_values
        c->set_output(out_idx++, c->Vector(InferenceContext::kUnknownDim));
      }
      ShapeHandle shape_v = c->Vector(2);
      for (int i = 0; i < top_paths; ++i) {  // decoded_shape
        c->set_output(out_idx++, shape_v);
      }
      for (int i = 0; i < top_paths; ++i) {  // alignment_indices
        c->set_output(out_idx++, c->Matrix(InferenceContext::kUnknownDim, 2));
      }
      for (int i = 0; i < top_paths; ++i) {  // alignment_values
        c->set_output(out_idx++, c->Vector(InferenceContext::kUnknownDim));
      }
      for (int i = 0; i < top_paths; ++i) {  // alignment_shape
        c->set_output(out_idx++, shape_v);
      }
      c->set_output(out_idx++, c->Matrix(batch_size, top_paths)); // log_probability
      return Status::OK();
    });
