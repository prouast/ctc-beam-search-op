"""Tests for ctc_beam_search_uncoll ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.platform import test
try:
  from tensorflow_ctc_beam_search_uncoll.python.ops.ctc_beam_search_uncoll_ops import ctc_beam_search_uncoll
except ImportError:
  from ctc_beam_search_uncoll_ops import ctc_beam_search_uncoll


class CTCBeamSearchUncollTest(test.TestCase):

  def testCTCBeamSearchUncoll(self):
    with self.test_session():
      self.assertAllClose(
          ctc_beam_search_uncoll([[1, 2], [3, 4]]), np.array([[1, 0], [0, 0]]))


if __name__ == '__main__':
  test.main()
