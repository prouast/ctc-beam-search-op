"""Tests for ctc_beam_search_u_decoder ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math

from tensorflow.python.platform import test
try:
    from tensorflow_ctc_beam_search_u_decoder.python.ops.ctc_beam_search_u_decoder_ops import ctc_beam_search_u_decoder
except ImportError:
    from ctc_beam_search_u_decoder_ops import ctc_beam_search_u_decoder


class CTCBeamSearchUDecoderTest(test.TestCase):

    def testCTCBeamSearchUDecoderV1(self):

        # Inputs have shape [max_time, batch_size, num_classes]
        # here: [2, 1, 2]
        # algorithm works with logits
        test_inputs = np.log(
            [[[0.1, 0.1, 0.8]],
             [[0.1, 0.2, 0.7]],
             [[0.2, 0.5, 0.3]],
             [[0.3, 0.4, 0.3]],
             [[0.1, 0.3, 0.6]],
             [[0.4, 0.3, 0.3]],
             [[0.6, 0.2, 0.2]],
             [[0.8, 0.1, 0.1]]])
        # Correct results:
        # Indices [top_paths, batch, time]
        correct_decoded_indices = np.array(
            [[[0, 0], [0, 1]],
             [[0, 0], [0, 1], [0, 2]],
             [[0, 0], [0, 1], [0, 2]]])
        correct_decoded_uncoll_indices = np.array(
            [[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7]],
             [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7]],
             [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7]]])
        # Values
        correct_decoded_values = np.array([[1, 0], [1, 1, 0], [1, 0, 0]])
        correct_decoded_uncoll_values = np.array([
            [-1, -1, 1, 1, -1, 0, 0, 0],
            [-1, -1, 1, 1, -1, 1, 0, 0],
            [-1, -1, 1, 1, -1, 0, -1, 0]])
        # Shape [batch_size, max_shape]
        correct_decoded_shape = np.array([[1, 2], [1, 3], [1, 3]])
        correct_decoded_uncoll_shape = np.array([[1, 8], [1, 8], [1, 8]])
        # Log probs [batch_size, top_paths]
        correct_log_probs = np.array([[-2.0580008, -2.7012699, -3.620572]])

        with self.test_session():
            self.assertAllClose(
                a=ctc_beam_search_u_decoder(inputs=test_inputs,
                    sequence_length=[8], beam_width=3, blank_index=2,
                    top_paths=3, merge_repeated=False)[0],
                b=correct_decoded_indices)
            self.assertAllClose(
                a=ctc_beam_search_u_decoder(inputs=test_inputs,
                    sequence_length=[8], beam_width=3, blank_index=2,
                    top_paths=3, merge_repeated=False)[1],
                b=correct_decoded_values)
            self.assertAllClose(
                a=ctc_beam_search_u_decoder(inputs=test_inputs,
                    sequence_length=[8], beam_width=3, blank_index=2,
                    top_paths=3, merge_repeated=False)[2],
                b=correct_decoded_shape)
            self.assertAllClose(
                a=ctc_beam_search_u_decoder(inputs=test_inputs,
                    sequence_length=[8], beam_width=3, blank_index=2,
                    top_paths=3, merge_repeated=False)[3],
                b=correct_decoded_uncoll_indices)
            self.assertAllClose(
                a=ctc_beam_search_u_decoder(inputs=test_inputs,
                    sequence_length=[8], beam_width=3, blank_index=2,
                    top_paths=3, merge_repeated=False)[4],
                b=correct_decoded_uncoll_values)
            self.assertAllClose(
                a=ctc_beam_search_u_decoder(inputs=test_inputs,
                    sequence_length=[8], beam_width=3, blank_index=2,
                    top_paths=3, merge_repeated=False)[5],
                b=correct_decoded_uncoll_shape)
            self.assertAllClose(
                a=ctc_beam_search_u_decoder(inputs=test_inputs,
                    sequence_length=[8], beam_width=3, blank_index=2,
                    top_paths=3, merge_repeated=False)[6],
                b=correct_log_probs)

    def testCTCBeamSearchUDecoderV2(self):

        # Inputs have shape [max_time, batch_size, num_classes]
        # here: [2, 1, 2]
        # algorithm works with logits
        test_inputs = np.log(
            [[[0.1, 0.1, 0.8]],
             [[0.1, 0.2, 0.7]],
             [[0.2, 0.5, 0.3]],
             [[0.3, 0.4, 0.3]],
             [[0.1, 0.3, 0.6]],
             [[0.4, 0.3, 0.3]],
             [[0.6, 0.2, 0.2]],
             [[0.8, 0.1, 0.1]]])
        # Correct results:
        # Indices [top_paths, batch, time]
        correct_decoded_indices = np.array(
            [[[0, 0], [0, 1]],
             [[0, 0], [0, 1], [0, 2]],
             [[0, 0], [0, 1], [0, 2]],
             [[0, 0], [0, 1], [0, 2], [0, 3]],
             [[0, 0], [0, 1], [0, 2]]])
        correct_decoded_uncoll_indices = np.array(
            [[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7]],
             [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7]],
             [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7]],
             [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7]],
             [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7]]])
        # Values
        correct_decoded_values = np.array(
            [[1, 0], [0, 1, 0], [1, 1, 0], [1, 0, 1, 0], [1, 0, 0]])
        correct_decoded_uncoll_values = np.array([
            [-1, -1, 1, 1, -1, 0, 0, 0],
            [-1, -1, 0, 1, -1, 0, 0, 0],
            [-1, -1, 1, 1, -1, 1, 0, 0],
            [-1, -1, 1, 0, -1, 1, 0, 0],
            [-1, -1, 1, 1, -1, 0, -1, 0]])
        # Shape [batch_size, max_shape]
        correct_decoded_shape = np.array([
            [1, 2], [1, 3], [1, 3], [1, 4], [1, 3]])
        correct_decoded_uncoll_shape = np.array([
            [1, 8], [1, 8], [1, 8], [1, 8], [1, 8]])
        # Log probs [batch_size, top_paths]
        correct_log_probs = np.array([
            [-1.7970481, -2.0828815, -2.1167731, -2.3315485, -2.5869188]])

        with self.test_session():
            self.assertAllClose(
                a=ctc_beam_search_u_decoder(inputs=test_inputs,
                    sequence_length=[8], beam_width=10, blank_index=2,
                    top_paths=5, merge_repeated=False)[0],
                b=correct_decoded_indices)
            self.assertAllClose(
                a=ctc_beam_search_u_decoder(inputs=test_inputs,
                    sequence_length=[8], beam_width=10, blank_index=2,
                    top_paths=5, merge_repeated=False)[1],
                b=correct_decoded_values)
            self.assertAllClose(
                a=ctc_beam_search_u_decoder(inputs=test_inputs,
                    sequence_length=[8], beam_width=10, blank_index=2,
                    top_paths=5, merge_repeated=False)[2],
                b=correct_decoded_shape)
            self.assertAllClose(
                a=ctc_beam_search_u_decoder(inputs=test_inputs,
                    sequence_length=[8], beam_width=10, blank_index=2,
                    top_paths=5, merge_repeated=False)[3],
                b=correct_decoded_uncoll_indices)
            self.assertAllClose(
                a=ctc_beam_search_u_decoder(inputs=test_inputs,
                    sequence_length=[8], beam_width=10, blank_index=2,
                    top_paths=5, merge_repeated=False)[4],
                b=correct_decoded_uncoll_values)
            self.assertAllClose(
                a=ctc_beam_search_u_decoder(inputs=test_inputs,
                    sequence_length=[8], beam_width=10, blank_index=2,
                    top_paths=5, merge_repeated=False)[5],
                b=correct_decoded_uncoll_shape)
            self.assertAllClose(
                a=ctc_beam_search_u_decoder(inputs=test_inputs,
                    sequence_length=[8], beam_width=10, blank_index=2,
                    top_paths=5, merge_repeated=False)[6],
                b=correct_log_probs)


if __name__ == '__main__':
    test.main()
