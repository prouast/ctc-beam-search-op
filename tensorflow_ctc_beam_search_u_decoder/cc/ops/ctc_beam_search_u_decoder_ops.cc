#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

// TODO Add shape assertions like
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/ctc_ops.cc

REGISTER_OP("CTCBeamSearchUDecoder")
    .Input("inputs: T")
    .Input("sequence_length: int32")
    .Attr("beam_width: int >= 1")
    .Attr("merge_repeated: bool = true")
    .Attr("blank_index: int = 0")
    .Attr("top_paths: int >= 1")
    .Attr("def_val: int = 0") // TODO do we need def_val?
    .Output("decoded_c: int32")
    .Output("decoded_u: int32")
    .Attr("T: {float, double} = DT_FLOAT")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(0));
      return Status::OK();
    });
