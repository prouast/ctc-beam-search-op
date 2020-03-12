"""Use ctc_ext_beam_search_decoder ops in python."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

ctc_ext_beam_search_decoder_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_ctc_ext_beam_search_decoder_ops.so'))
ctc_ext_beam_search_decoder = ctc_ext_beam_search_decoder_ops.ctc_ext_beam_search_decoder
