# TensorFlow CTC Beam Search Decoder that keeps track of best alignment for each beam
This is a custom version of the [`tf.nn.ctc_beam_search_decoder`](https://www.tensorflow.org/api_docs/python/tf/nn/ctc_beam_search_decoder).
During the beam search, it keeps track of the most probable alignment for each beam, which includes blank labels.
This makes it possible to estimate at which point in the sequence a label might be situated.

This op was implemented for CPU only as a c++ kernel following the [guide](https://github.com/tensorflow/custom-op) provided by the TensorFlow authors.
Refer to the guide for instructions on how to build and test the Op using Docker for Windows or Linux.

### Args

- `inputs`: 3-D `float` `Tensor`, size `[max_time, batch_size, num_classes]`. Input logits.
- `sequence_length`: 1-D `int32` vector containing sequence lengths, having size `[batch_size]`.
- `beam_width`: An int scalar >= 1 (beam search beam width).
- `top_paths`: An int scalar >= 1, <= beam_width (controls output size).
- `merge_repeated`: A boolean indicating whether repeated sequence elements are merged in regular output.
- `blank_index`: An int < `num_classes` indicating which index in the inputs corresponds to blank labels.
- `blank_label`: An int indicating the label that should be used for blanks in alignment.

### Outputs

A tuple `(decoded, alignment, log_probability)` where

- `decoded`: A list of length `top_paths`, where `decoded[j]` is a `SparseTensor` containing the decoded outputs:
  * `decoded[j].indices`: Indices matrix `[total_decoded_outputs[j], 2]`; The rows store: `[batch, time]`.
  * `decoded[j].values`: Values vector, size `[total_decoded_outputs[j]]`. The vector stores the decoded classes for beam `j`.
  * `decoded[j].shape`: Shape vector, size `(2)`. The shape values are: `[batch_size, max_decoded_length[j]]`.
- `alignment`: A list of length `top_paths`, where `decoded[j]` is a `SparseTensor` containing the alignments of decoded outputs:
  * `alignment[j].indices`: Indices matrix `[sequence_length[j], 2]`; The rows store: `[batch, time]`.
  * `alignment[j].values`: Values vector, size `[sequence_length[j]]`. The vector stores the alignment for beam `j`.
  * `alignment[j].shape`: Shape vector, size `(2)`. The shape values are: `[batch_size, sequence_length[j]]`.
- `log_probability`: A float matrix `[batch_size, top_paths]` containing sequence log-probabilities.
