#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

// TODO: SetShapeFn
REGISTER_OP("CTCBeamSearchUncoll")
    .Input("inputs: int32")
    .Output("zeroed: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });
    //.Input("inputs: float")
    //.Input("sequence_length: int32")
    //.Input("beam_width: int32")
    //.Input("blank: int32")
    //.Input("def_val: int32")
    //.Output("decoded_c: int32")
    //.Output("decoded_u: int32")
    //.SetShapeFn(::tensorflow::shape_inference::UnknownShape);
