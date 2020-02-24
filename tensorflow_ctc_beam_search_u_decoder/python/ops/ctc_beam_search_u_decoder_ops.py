"""Use ctc_beam_search_u_decoder ops in python."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

ctc_beam_search_u_decoder_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_ctc_beam_search_u_decoder_ops.so'))
ctc_beam_search_u_decoder = ctc_beam_search_u_decoder_ops.ctc_beam_search_u_decoder
