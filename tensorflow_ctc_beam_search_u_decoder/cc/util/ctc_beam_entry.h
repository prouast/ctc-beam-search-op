/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <algorithm>
#include <memory>
#include <vector>
#include <iostream>
#include <sstream>
#include <queue>
#include "ctc_loss_util.h"
#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace ctc {
namespace ctc_beam_search {

struct EmptyBeamState {};

template <typename T>
struct BeamProbability {
  BeamProbability()
      : total(kLogZero<T>()), blank(kLogZero<T>()), label(kLogZero<T>()) {}
  void Reset() {
    total = kLogZero<T>();
    blank = kLogZero<T>();
    label = kLogZero<T>();
  }
  T total;
  T blank;
  T label;
};

template <typename T>
struct BeamUncollCandidate {
  BeamUncollCandidate() : prob(kLogZero<T>()) {}
  BeamUncollCandidate(std::vector<int> l, T p) : prob(p), label_seq(l) {}
  T prob;
  std::vector<int> label_seq;
  BeamUncollCandidate<T>* clone() const {
    return new BeamUncollCandidate<T>(*this);
  }
  std::string Print() {
    std::stringstream ss;
    ss << "[" << prob << ": [";
    for (size_t i = 0; i < label_seq.size(); ++i) {
      if (i != 0)
        ss << ",";
      ss << label_seq[i];
    }
    ss << "]]";
    return ss.str();
  }
};

template <class T>
class BeamUncollCandidateComparer {
 public:
  virtual ~BeamUncollCandidateComparer() {}
  virtual bool inline operator()(const BeamUncollCandidate<T>* a,
                                 const BeamUncollCandidate<T>* b) const {
    return a->prob < b->prob;
  }
};

template <typename T>
struct BeamUncoll {
  BeamUncoll() : cands_blank(), cands_nblank() {}
  std::priority_queue<BeamUncollCandidate<T>*,
    std::vector<BeamUncollCandidate<T>*>,
    BeamUncollCandidateComparer<T>> cands_blank;
  std::priority_queue<BeamUncollCandidate<T>*,
    std::vector<BeamUncollCandidate<T>*>,
    BeamUncollCandidateComparer<T>> cands_nblank;
  void Reset() {
    while (!cands_blank.empty()) {
      cands_blank.pop();
    }
    while (!cands_nblank.empty()) {
      cands_nblank.pop();
    }
  }
  BeamUncollCandidate<T>* GetBlank() {
    BeamUncollCandidate<T>* result = cands_blank.top();
    return result;
  }
  BeamUncollCandidate<T>* GetNBlank() {
    BeamUncollCandidate<T>* result = cands_nblank.top();
    return result;
  }
};

template <class T, class CTCBeamState>
class BeamRoot;

template <class T, class CTCBeamState = EmptyBeamState>
struct BeamEntry {
  // BeamRoot<CTCBeamState>::AddEntry() serves as the factory method.
  friend BeamEntry<T, CTCBeamState>* BeamRoot<T, CTCBeamState>::AddEntry(
      BeamEntry<T, CTCBeamState>* p, int l);
  inline bool Active() const { return newp.total != kLogZero<T>(); }
  inline bool New() const { return oldp.total == kLogZero<T>(); }
  // Return the child at the given index, or construct a new one in-place if
  // none was found.
  BeamEntry<T, CTCBeamState>& GetChild(int ind) {
    auto entry = children.emplace(ind, nullptr);
    auto& child_entry = entry.first->second;
    // If this is a new child, populate the BeamEntry<CTCBeamState>*.
    if (entry.second) {
      child_entry = beam_root->AddEntry(this, ind);
    }
    return *child_entry;
  }
  std::vector<int> LabelSeq(bool merge_repeated) const {
    std::vector<int> labels;
    int prev_label = -1;
    const BeamEntry<T, CTCBeamState>* c = this;
    while (c->parent != nullptr) {  // Checking c->parent to skip root leaf.
      if (!merge_repeated || c->label != prev_label) {
        labels.push_back(c->label);
      }
      prev_label = c->label;
      c = c->parent;
    }
    std::reverse(labels.begin(), labels.end());
    return labels;
  }
  std::vector<int> LabelSeqUncoll() {
    std::vector<int> labels;
    if (!new_cands.cands_blank.empty() && !new_cands.cands_nblank.empty()) {
      if (new_cands.GetBlank()->prob > new_cands.GetNBlank()->prob)
        labels = new_cands.GetBlank()->label_seq;
      else
        labels = new_cands.GetNBlank()->label_seq;
    } else if (!new_cands.cands_blank.empty()) {
      labels = new_cands.GetBlank()->label_seq;
    } else if (!new_cands.cands_nblank.empty()) {
      labels = new_cands.GetNBlank()->label_seq;
    } else {
      std::cout << "No label seq available" << std::endl;
    }
    return labels;
  }
  void Print(bool new_, bool full) {
    std::vector<int> label_seq = LabelSeq(false);
    std::stringstream ss;
    for (size_t i = 0; i < label_seq.size(); ++i) {
      if (i != 0)
        ss << ",";
      ss << label_seq[i];
    }
    std::string s = ss.str();
    if (full) {
      std::cout << "====================" << std::endl;
    }
    if (new_) {
      std::cout << "[label=" << s << "; " << "newp_blank=" << newp.blank << "; " << "newp_label=" << newp.label << "; " << "newp_total=" << newp.total << "]" << std::endl;
    } else {
      std::cout << "[label=" << s << "; " << "oldp_blank=" << oldp.blank << "; " << "oldp_label=" << oldp.label << "; " << "oldp_total=" << oldp.total << "]" << std::endl;
    }
    if (full) {
      BeamUncoll<T> cands = new_ ? new_cands : old_cands;
      std::cout << "------- top blank ------" << std::endl;
      if (!cands.cands_blank.empty()) {
        std::cout << cands.GetBlank()->Print() << std::endl;
      }
      std::cout << "------- top nblank ------" << std::endl;
      if (!cands.cands_nblank.empty()) {
        std::cout << cands.GetNBlank()->Print() << std::endl;
      }
      std::cout << "====================" << std::endl;
    }
  }

  BeamEntry<T, CTCBeamState>* parent;
  int label;
  BeamUncoll<T> old_cands;
  BeamUncoll<T> new_cands;

  // Add a new UncollCandidate
  void AddUncollCandidate(BeamEntry<T>* entry, bool from_blank, bool to_blank, int l, T p) {
    // Get old UncollCandidate
    BeamUncollCandidate<T>* old_uncoll = nullptr;
    if (from_blank && !entry->old_cands.cands_blank.empty())
      old_uncoll = entry->old_cands.GetBlank();
    else if (!from_blank && !entry->old_cands.cands_nblank.empty())
      old_uncoll = entry->old_cands.GetNBlank();
    // Get old probability
    std::vector<int> old_label_seq;
    T old_prob;
    if (old_uncoll == nullptr) {
      // If no previous UncollCandidate exists, we will start a new one with
      //  probability 0 if dealing with a new Beam in the first or second time
      //  step, otherwise with kLogZero
      if (parent == nullptr || parent->parent == nullptr && New()) {
        old_prob = from_blank ? 0 : kLogZero<T>();
      } else {
        old_prob = kLogZero<T>();
      }
    } else {
      // Build on the previous UncollCandidate
      old_prob = old_uncoll->prob;
      old_label_seq = old_uncoll->label_seq;
    }
    // Concat the label
    std::vector<int> new_label_seq(old_label_seq);
    new_label_seq.push_back(l);
    // Add the probabilities since we are using log probabilities
    T new_prob = old_prob + p;
    // New BeamUncollCandidate
    BeamUncollCandidate<T>* new_uncoll = new BeamUncollCandidate<T>(
      new_label_seq, new_prob);
    // Add new BeamUncollCandidate
    if (to_blank) {
      new_cands.cands_blank.push(new_uncoll);
    } else {
      new_cands.cands_nblank.push(new_uncoll);
    }
  }

  // All instances of child BeamEntry are owned by *beam_root.
  gtl::FlatMap<int, BeamEntry<T, CTCBeamState>*> children;
  BeamProbability<T> oldp;
  BeamProbability<T> newp;
  CTCBeamState state;

 private:
  // Constructor giving parent, label, and the beam_root.
  // The object pointed to by p cannot be copied and should not be moved,
  // otherwise parent will become invalid.
  // This private constructor is only called through the factory method
  // BeamRoot<CTCBeamState>::AddEntry().
  BeamEntry(BeamEntry* p, int l, BeamRoot<T, CTCBeamState>* beam_root)
      : parent(p), label(l), beam_root(beam_root) {}
  BeamRoot<T, CTCBeamState>* beam_root;
  TF_DISALLOW_COPY_AND_ASSIGN(BeamEntry);
};

// This class owns all instances of BeamEntry.  This is used to avoid recursive
// destructor call during destruction.
template <class T, class CTCBeamState = EmptyBeamState>
class BeamRoot {
 public:
  BeamRoot(BeamEntry<T, CTCBeamState>* p, int l) {
    root_entry_ = AddEntry(p, l);
  }
  BeamRoot(const BeamRoot&) = delete;
  BeamRoot& operator=(const BeamRoot&) = delete;

  BeamEntry<T, CTCBeamState>* AddEntry(BeamEntry<T, CTCBeamState>* p, int l) {
    // Copy the UncollCandidate from parent if available
    BeamUncollCandidate<T>* buc_b = nullptr;
    BeamUncollCandidate<T>* buc_nb = nullptr;
    auto* new_entry = new BeamEntry<T, CTCBeamState>(p, l, this);
    beam_entries_.emplace_back(new_entry);
    return new_entry;
  }
  BeamEntry<T, CTCBeamState>* RootEntry() const { return root_entry_; }

 private:
  BeamEntry<T, CTCBeamState>* root_entry_ = nullptr;
  std::vector<std::unique_ptr<BeamEntry<T, CTCBeamState>>> beam_entries_;
};

// BeamComparer is the default beam comparer provided in CTCBeamSearch.
template <class T, class CTCBeamState = EmptyBeamState>
class BeamComparer {
 public:
  virtual ~BeamComparer() {}
  virtual bool inline operator()(const BeamEntry<T, CTCBeamState>* a,
                                 const BeamEntry<T, CTCBeamState>* b) const {
    return a->newp.total > b->newp.total;
  }
};

}  // namespace ctc_beam_search
}  // namespace ctc
}  // namespace tensorflow
