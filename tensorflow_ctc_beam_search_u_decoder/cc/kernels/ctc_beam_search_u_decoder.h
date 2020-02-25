// TODO

#include "ctc_u_decoder.h"
#include "ctc_beam_entry.h"

#include "third_party/eigen3/Eigen/Core"

namespace tensorflow {
namespace ctc {

template <typename T, typename CTCBeamState = ctc_beam_search::EmptyBeamState,
          typename CTCBeamComparer = ctc_beam_search::BeamComparer<T, CTCBeamState>>
class CTCBeamSearchUDecoder : public CTCUDecoder<T> {

  typedef ctc_beam_search::BeamEntry<T, CTCBeamState> BeamEntry;
  typedef ctc_beam_search::BeamRoot<T, CTCBeamState> BeamRoot;
  typedef ctc_beam_search::BeamProbability<T> BeamProbability;

  public:

    typedef BaseBeamScorer<T, CTCBeamState> DefaultBeamScorer;

    // Constructor
    CTCBeamSearchUDecoder(int num_classes, int blank_index, int beam_width,
                          BaseBeamScorer<T, CTCBeamState>* scorer,
                          int batch_size=1, bool merge_repeated = false)
        : CTCUDecoder<T>(num_classes, blank_index, batch_size, merge_repeated),
          beam_width_(beam_width), leaves_(beam_width),
          beam_scorer_(CHECK_NOTNULL(scorer)) {
      Reset();
    }

    ~CTCBeamSearchUDecoder() override {}

    // Run the hibernating beam search algorithm on the given input.
    Status Decode(const typename CTCUDecoder<T>::SequenceLength& seq_len,
                  const std::vector<typename CTCUDecoder<T>::Input>& input,
                  std::vector<typename CTCUDecoder<T>::Output>* output,
                  typename CTCUDecoder<T>::ScoreOutput* scores) override;

    // Calculate the next step of the beam search and update the internal state.
    template <typename Vector>
    void Step(const Vector& log_input_t);

    template <typename Vector>
    T GetTopK(const int K, const Vector& input, std::vector<T>* top_k_logits,
              std::vector<int>* top_k_indices);

    // Retrieve the beam scorer instance used during decoding.
    BaseBeamScorer<T, CTCBeamState>* GetBeamScorer() const {
      return beam_scorer_;
    }

    // Reset the beam search
    void Reset();

    // Extract the top n paths at current time step
    Status TopPaths(int n, std::vector<std::vector<int>>* paths,
                    std::vector<T>* log_probs, bool merge_repeated) const;

  private:

    int beam_width_;
    gtl::TopN<BeamEntry*, CTCBeamComparer> leaves_;
    std::unique_ptr<BeamRoot> beam_root_;
    BaseBeamScorer<T, CTCBeamState>* beam_scorer_;

    TF_DISALLOW_COPY_AND_ASSIGN(CTCBeamSearchUDecoder);
};


// TODO understand/implement methods


} // namespace ctc
} // namespace tensorflow
