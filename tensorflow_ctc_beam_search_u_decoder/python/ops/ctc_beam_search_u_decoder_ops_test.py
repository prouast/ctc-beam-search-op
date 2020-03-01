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

    def testCTCBeamSearchUDecoder(self):

        # Inputs have shape [max_time, batch_size, num_classes]
        # here: [2, 1, 2]
        # algorithm works with logits
        test_inputs = np.log([[[0.3, 0.7]], [[0.4, 0.6]]])

        with self.test_session():
            self.assertAllClose(
                a=ctc_beam_search_u_decoder(inputs=test_inputs,
                    sequence_length=[2], beam_width=5, blank_index=1)[0],
                b=np.array([[[5, 0]], [[0, 0]]]))


if __name__ == '__main__':
    test.main()
