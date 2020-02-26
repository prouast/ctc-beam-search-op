//

#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace ctc {

template <class T>
class CTCUDecoder {
  public:
    typedef Eigen::Map<const Eigen::ArrayXi> SequenceLength;
    typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> Input;
    typedef std::vector<std::vector<int>> Output;
    typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> ScoreOutput;

    CTCUDecoder(int num_classes, int blank_index, int batch_size, bool merge_repeated)
      : num_classes_(num_classes), blank_index_(blank_index),
        batch_size_(batch_size), merge_repeated_(merge_repeated) {}

    virtual ~CTCUDecoder() {}

    int batch_size() { return batch_size_; }
    int num_classes() { return num_classes_; }

  protected:
    int num_classes_;
    int blank_index_;
    int batch_size_;
    bool merge_repeated_;
};

} // namespace ctc
} // namespace tensorflow
