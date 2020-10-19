"""Tests for ctc_ext_beam_search_decoder ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import psutil

from tensorflow.python.platform import test
try:
    from tensorflow_ctc_ext_beam_search_decoder.python.ops.ctc_ext_beam_search_decoder_ops import ctc_ext_beam_search_decoder
except ImportError:
    from ctc_ext_beam_search_decoder_ops import ctc_ext_beam_search_decoder


class CTCExtBeamSearchDecoderTest(test.TestCase):

    def testCTCExtBeamSearchDecoderPaper(self):
        # Example from paper
        # Inputs have shape [max_time, batch_size, num_classes]
        # here: [8, 1, 2]
        # algorithm works with logits
        logits = np.log(
            [[[0.3, 0.5, 0.2]],
             [[0.25, 0.6, 0.15]],
             [[0.6, 0.2, 0.2]],
             [[0.4, 0.35, 0.25]],
             [[0.5, 0.4, 0.1]],
             [[0.3, 0.3, 0.4]],
             [[0.1, 0.2, 0.7]],
             [[0.2, 0.3, 0.5]]])
        # Correct results:
        # Indices [top_paths, batch, time]
        correct_decoded_indices = np.array(
            [[[0, 0], [0, 1], [0, 2]],
             [[0, 0], [0, 1], [0, 2], [0, 3]],
             [[0, 0], [0, 1]],
             [[0, 0], [0, 1], [0, 2], [0, 3]],
             [[0, 0], [0, 1], [0, 2]]])
        correct_alignment_indices = np.array(
            [[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7]],
             [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7]],
             [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7]],
             [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7]],
             [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7]]])
        # Values
        correct_decoded_values = np.array(
            [[1, 1, 2], [1, 2, 1, 2], [1, 2], [1, 1, 2, 1], [1, 2, 1]])
        correct_alignment_values = np.array([
            [1, 1, 0, 1, 0, 2, 2, 2],
            [1, 1, 0, 2, 1, 2, 2, 2],
            [1, 1, 0, 0, 0, 2, 2, 2],
            [1, 1, 0, 1, 0, 2, 2, 1],
            [1, 1, 0, 0, 0, 2, 2, 1]])
        # Shape [batch_size, max_shape]
        correct_decoded_shape = np.array([
            [1, 3], [1, 4], [1, 2], [1, 4], [1, 3]])
        correct_alignment_shape = np.array([
            [1, 8], [1, 8], [1, 8], [1, 8], [1, 8]])
        # Log probs [batch_size, top_paths]
        correct_log_probs = np.array([
            [-2.0613022, -2.1155741, -2.713197, -2.8770373, -2.9212725]])
        with self.test_session():
            self.assertAllClose(
                a=ctc_ext_beam_search_decoder(inputs=logits,
                    sequence_length=[8], beam_width=10, blank_index=0,
                    top_paths=5, blank_label=0, merge_repeated=False)[0],
                b=correct_decoded_indices)
            self.assertAllClose(
                a=ctc_ext_beam_search_decoder(inputs=logits,
                    sequence_length=[8], beam_width=10, blank_index=0,
                    top_paths=5, blank_label=0, merge_repeated=False)[1],
                b=correct_decoded_values)
            self.assertAllClose(
                a=ctc_ext_beam_search_decoder(inputs=logits,
                    sequence_length=[8], beam_width=10, blank_index=0,
                    top_paths=5, blank_label=0, merge_repeated=False)[2],
                b=correct_decoded_shape)
            self.assertAllClose(
                a=ctc_ext_beam_search_decoder(inputs=logits,
                    sequence_length=[8], beam_width=10, blank_index=0,
                    top_paths=5, blank_label=0, merge_repeated=False)[3],
                b=correct_alignment_indices)
            self.assertAllClose(
                a=ctc_ext_beam_search_decoder(inputs=logits,
                    sequence_length=[8], beam_width=10, blank_index=0,
                    top_paths=5, blank_label=0, merge_repeated=False)[4],
                b=correct_alignment_values)
            self.assertAllClose(
                a=ctc_ext_beam_search_decoder(inputs=logits,
                    sequence_length=[8], beam_width=10, blank_index=0,
                    top_paths=5, blank_label=0, merge_repeated=False)[5],
                b=correct_alignment_shape)
            self.assertAllClose(
                a=ctc_ext_beam_search_decoder(inputs=logits,
                    sequence_length=[8], beam_width=10, blank_index=0,
                    top_paths=5, blank_label=0, merge_repeated=False)[6],
                b=correct_log_probs)

    def testCTCExtBeamSearchDecoderMemLeak(self):
      with self.test_session():
        test_inputs = np.log(
          [[[0.3, 0.5, 0.2]],
           [[0.25, 0.6, 0.15]],
           [[0.6, 0.2, 0.2]],
           [[0.4, 0.35, 0.25]],
           [[0.5, 0.4, 0.1]],
           [[0.3, 0.3, 0.4]],
           [[0.1, 0.2, 0.7]],
           [[0.2, 0.3, 0.5]]])
        ctc_ext_beam_search_decoder(inputs=test_inputs,
          sequence_length=[8], beam_width=10, blank_index=0,
          top_paths=5, blank_label=0, merge_repeated=False)
        mem_after_1_it = psutil.virtual_memory().used
        for i in range(999):
          ctc_ext_beam_search_decoder(inputs=test_inputs,
            sequence_length=[8], beam_width=10, blank_index=0,
            top_paths=5, blank_label=0, merge_repeated=False)
        mem_after_1000_it = psutil.virtual_memory().used
        print("Leak: {}".format(mem_after_1000_it-mem_after_1_it))
        self.assertLessEqual(mem_after_1000_it, mem_after_1_it)

if __name__ == '__main__':
    test.main()
