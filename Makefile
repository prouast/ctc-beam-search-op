CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python

CTC_BEAM_SEARCH_SRCS = $(wildcard tensorflow_ctc_beam_search_u_decoder/cc/kernels/*.cc) $(wildcard tensorflow_ctc_beam_search_u_decoder/cc/kernels/*.h) $(wildcard tensorflow_ctc_beam_search_u_decoder/cc/ops/*.cc)

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11
LDFLAGS = -shared ${TF_LFLAGS}

CTC_BEAM_SEARCH_TARGET_LIB = tensorflow_ctc_beam_search_u_decoder/python/ops/_ctc_beam_search_u_decoder_ops.so

# ctc_beam_search_u_decoder op for CPU
ctc_beam_search_u_decoder_op: $(CTC_BEAM_SEARCH_TARGET_LIB)

$(CTC_BEAM_SEARCH_TARGET_LIB): $(CTC_BEAM_SEARCH_SRCS)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}

ctc_beam_search_u_decoder_test: tensorflow_ctc_beam_search_u_decoder/python/ops/ctc_beam_search_u_decoder_ops_test.py tensorflow_ctc_beam_search_u_decoder/python/ops/ctc_beam_search_u_decoder_ops.py $(CTC_BEAM_SEARCH_TARGET_LIB)
	$(PYTHON_BIN_PATH) tensorflow_ctc_beam_search_u_decoder/python/ops/ctc_beam_search_u_decoder_ops_test.py

ctc_beam_search_u_decoder_pip_pkg: $(CTC_BEAM_SEARCH_TARGET_LIB)
	./build_pip_pkg.sh make artifacts

clean:
	rm -f $(CTC_BEAM_SEARCH_TARGET_LIB)
