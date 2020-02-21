#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

// TODO Add shape assertions

REGISTER_OP("CTCBeamSearchUncoll")
    .Input("inputs: float")
    .Attr("sequence_length: int")
    .Attr("beam_width: int")
    .Attr("blank: int=0")
    .Attr("def_val: int=0")
    .Output("decoded_c: int32")
    .Output("decoded_u: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(0));
      return Status::OK();
    });
