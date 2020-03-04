#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

// TODO Add shape assertions like
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/ctc_ops.cc

REGISTER_OP("CTCBeamSearchUDecoder")
    .Input("inputs: T")
    .Input("sequence_length: int32")
    .Attr("beam_width: int >= 1")
    .Attr("top_paths: int >= 1")
    .Attr("merge_repeated: bool = False")
    .Attr("blank_index: int = 0")
    .Output("decoded_indices: top_paths * int64")
    .Output("decoded_values: top_paths * int64")
    .Output("decoded_shape: top_paths * int64")
    .Output("decoded_uncoll_indices: top_paths * int64")
    .Output("decoded_uncoll_values: top_paths * int64")
    .Output("decoded_uncoll_shape: top_paths * int64")
    .Output("log_probability: T")
    .Attr("T: {float, double} = DT_FLOAT")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(0));
      return Status::OK();
    });
