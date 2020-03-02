// TODO

#include <vector>

#include "ctc_beam_scorer.h"
#include "ctc_u_decoder.h"

#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/top_n.h"

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
                          int batch_size=1, bool merge_repeated=false)
        : CTCUDecoder<T>(num_classes, blank_index, batch_size, merge_repeated),
          beam_width_(beam_width), leaves_(beam_width),
          beam_scorer_(CHECK_NOTNULL(scorer)) {
      Reset();
    }

    ~CTCBeamSearchUDecoder() override {}

    // Calculate the next step of the beam search and update the internal state.
    template <typename Vector>
    void Step(const Vector& log_input_t);

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

// Run one sequence step of beam search
template <typename T, typename CTCBeamState, typename CTCBeamComparer>
template <typename Vector>
void CTCBeamSearchUDecoder<T, CTCBeamState, CTCBeamComparer>::Step(
    const Vector& raw_input) {
  // Number of character classes to consider in each step.
  const int max_classes = this->num_classes_ - 1;
  // Get max coefficient and remove it from raw_input later.
  T max_coeff = raw_input.maxCoeff();
  // Get normalization term of softmax: log(sum(exp(logit[j]-max_coeff))).
  T logsumexp = T(0.0);
  for (int j = 0; j < raw_input.size(); ++j) {
    logsumexp += Eigen::numext::exp(raw_input(j) - max_coeff);
  }
  logsumexp = Eigen::numext::log(logsumexp);
  // Final normalization offset to get correct log probabilities.
  T norm_offset = max_coeff + logsumexp;

  // Extract the beams sorted in decreasing new probability
  CHECK_EQ(this->num_classes_, raw_input.size());
  std::unique_ptr<std::vector<BeamEntry*>> branches(leaves_.Extract());
  leaves_.Reset();

  for (BeamEntry* b : *branches) {
    // P(.. @ t) becomes the new P(.. @ t-1)
    b->oldp = b->newp;
  }

  // A. Enter same BeamEntries into new proposals if not already there
  for (BeamEntry* b : *branches) {
    if (b->parent != nullptr) {  // Non-empty beam (not the root)
      if (b->parent->Active()) {
        if (b->label == b->parent->label) {
          // If last two sequence characters are identical:
          //   Plabel(l=acc @ t=6) = (Plabel(l=acc @ t=5) + Pblank(l=ac @ t=5))
          //                         * P(c @ 6)
          b->newp.label = LogSumExp(b->newp.label,
            beam_scorer_->GetStateExpansionScore(b->state, b->parent->oldp.blank))
            + raw_input(b->label) - norm_offset;
          b->AddUncollCandidate(true, false, b->label, raw_input(b->label));
        } else {
          // If the last two sequence characters are not identical:
          //   Plabel(l=abc @ t=6) = (Plabel(l=abc @ t=5) + P(l=ab @ t=5))
          //                         * P(c @ 6)
          b->newp.label = LogSumExp(b->newp.label,
            beam_scorer_->GetStateExpansionScore(b->state, b->parent->oldp.total))
            + raw_input(b->label) - norm_offset;
          b->AddUncollCandidate(true, false, b->label, raw_input(b->label));
          b->AddUncollCandidate(false, false, b->label, raw_input(b->label));
        }
      } else {
        // Plabel(l=abc @ t=6) *= P(c @ 6)
        b->newp.label += raw_input(b->label) - norm_offset;
        b->AddUncollCandidate(true, false, b->label, raw_input(b->label));
        b->AddUncollCandidate(false, false, b->label, raw_input(b->label));
      }
      ////
      //if (b->parent->Active()) {
        // If last two sequence characters are identical:
        //   Plabel(l=acc @ t=6) = (Plabel(l=acc @ t=5)
        //                          + Pblank(l=ac @ t=5))
        // else:
        //   Plabel(l=abc @ t=6) = (Plabel(l=abc @ t=5)
        //                          + P(l=ab @ t=5))
        //T previous = (b->label == b->parent->label) ? b->parent->oldp.blank
        //                                            : b->parent->oldp.total;
        //b->newp.label = LogSumExp(b->newp.label,
          //beam_scorer_->GetStateExpansionScore(b->state, previous));
      //}
      // Plabel(l=abc @ t=6) *= P(c @ 6)
      //b->newp.label += raw_input(b->label) - norm_offset;
      //b->AddUncollCandidate(, false, b->label, b->newp.label);
    }
    // Adding a blank to the entry
    // Pblank(l=abc @ t=6) = P(l=abc @ t=5) * P(- @ 6)
    b->newp.blank = b->oldp.total + raw_input(this->blank_index_) - norm_offset;
    //b->AddUncollCandidate(this->blank_index_, b->newp.blank, true);
    // Calculate p_total = p_label + p_blank
    // P(l=abc @ t=6) = Plabel(l=abc @ t=6) + Pblank(l=abc @ t=6)
    b->newp.total = LogSumExp(b->newp.blank, b->newp.label);
    // Push the entry back to the top paths list.
    // Note, this will always fill leaves back up in sorted order.
    //std::cout << "NewA:" << std::endl;
    //b->Print(true, true);
    leaves_.push(b);
  }

  // B. Extend BeamEntries with a non-blank event
  for (BeamEntry* b : *branches) {
    // A new leaf (represented by its BeamProbability) is a candidate
    // if its total probability is nonzero and either the beam list
    // isn't full, or the lowest probability entry in the beam has a
    // lower probability than the leaf.
    auto is_candidate = [this](const BeamProbability& prob) {
      return (prob.total > kLogZero<T>() &&
              (leaves_.size() < beam_width_ ||
               prob.total > leaves_.peek_bottom()->newp.total));
    };

    if (!is_candidate(b->oldp)) {
      continue;
    }

    for (int ind = 0; ind < max_classes; ind++) {
      const int label = ind;
      const T logit = raw_input(ind);
      // The new BeamEntry
      BeamEntry& c = b->GetChild(label);
      if (!c.Active()) {
        //   Pblank(l=abcd @ t=6) = 0
        c.newp.blank = kLogZero<T>();
        // If new child label is identical to beam label:
        //   Plabel(l=abcc @ t=6) = Pblank(l=abc @ t=5) * P(c @ 6)
        // Otherwise:
        //   Plabel(l=abcd @ t=6) = P(l=abc @ t=5) * P(d @ 6)
        beam_scorer_->ExpandState(b->state, b->label, &c.state, c.label);
        //T previous = (c.label == b->label) ? b->oldp.blank : b->oldp.total;
        //c.newp.label = logit - norm_offset +
        //               beam_scorer_->GetStateExpansionScore(c.state, previous);
        if (c.label == b->label) {
          c.newp.label = logit - norm_offset +
                         beam_scorer_->GetStateExpansionScore(c.state, b->oldp.blank);
          c.AddUncollCandidate(true, false, label, c.newp.label);
        } else {
          c.newp.label = logit - norm_offset +
                         beam_scorer_->GetStateExpansionScore(c.state, b->oldp.total);
          c.AddUncollCandidate(true, false, label, c.newp.label);
          c.AddUncollCandidate(false, false, label, c.newp.label);
        }
        // P(l=abcd @ t=6) = Plabel(l=abcd @ t=6)
        c.newp.total = c.newp.label;

        if (is_candidate(c.newp)) {
          // Before adding the new node to the beam, check if the beam
          // is already at maximum width.
          if (leaves_.size() == beam_width_) {
            // Bottom is no longer in the beam search.  Reset
            // its probability; signal it's no longer in the beam search.
            BeamEntry* bottom = leaves_.peek_bottom();
            bottom->newp.Reset();
          }
          //std::cout << "NewB:" << std::endl;
          //c.Print(true, true);
          leaves_.push(&c);
        } else {
          // Deactivate child.
          c.oldp.Reset();
          c.newp.Reset();
        }
      }
    }
  }
  std::cout << "EndStep:" << std::endl;
  // Resolve candidates in each beam
  std::unique_ptr<std::vector<BeamEntry*>> branches_(leaves_.ExtractNondestructive());
  for (BeamEntry* b : *branches) {
    // TODO uncoll and coll cands for each beam and set uncoll_blank and uncoll_nblank
    b->Print(true, true);
  }
}

template <typename T, typename CTCBeamState, typename CTCBeamComparer>
void CTCBeamSearchUDecoder<T, CTCBeamState, CTCBeamComparer>::Reset() {
  leaves_.Reset();

  // This beam root, and all of its children, will be in memory until
  // the next reset.
  beam_root_.reset(new BeamRoot(nullptr, -1));
  beam_root_->RootEntry()->newp.total = T(0.0);  // ln(1)
  beam_root_->RootEntry()->newp.blank = T(0.0);  // ln(1)

  // Add the root as the initial leaf.
  leaves_.push(beam_root_->RootEntry());

  // Call initialize state on the root object.
  beam_scorer_->InitializeState(&beam_root_->RootEntry()->state);
}

template <typename T, typename CTCBeamState, typename CTCBeamComparer>
Status CTCBeamSearchUDecoder<T, CTCBeamState, CTCBeamComparer>::TopPaths(
    int n, std::vector<std::vector<int>>* paths, std::vector<T>* log_probs,
    bool merge_repeated) const {
  CHECK_NOTNULL(paths)->clear();
  CHECK_NOTNULL(log_probs)->clear();
  if (n > beam_width_) {
    return errors::InvalidArgument("requested more paths than the beam width.");
  }
  if (n > leaves_.size()) {
    return errors::InvalidArgument(
        "Less leaves in the beam search than requested.");
  }

  gtl::TopN<BeamEntry*, CTCBeamComparer> top_branches(n);

  // O(beam_width_ * log(n)), space complexity is O(n)
  for (auto it = leaves_.unsorted_begin(); it != leaves_.unsorted_end(); ++it) {
    top_branches.push(*it);
  }
  // O(n * log(n))
  std::unique_ptr<std::vector<BeamEntry*>> branches(top_branches.Extract());

  for (int i = 0; i < n; ++i) {
    BeamEntry* e((*branches)[i]);
    paths->push_back(e->LabelSeq(merge_repeated));
    log_probs->push_back(e->newp.total);
  }
  return Status::OK();
}

} // namespace ctc
} // namespace tensorflow
