/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <utils/mcts/cpu/cpu_mcts_tree_node.h>
#include <limits>
namespace mindspore_rl {
namespace utils {
void CPUMonteCarloTreeNode::InitNode(int state_size, float *init_reward,
                                     int *action, float *prior) {
  // Initialize and allocate memory
  state_ = reinterpret_cast<float *>(AllocateMem(sizeof(float) * state_size));
  total_reward_ = reinterpret_cast<float *>(AllocateMem(sizeof(float)));
  explore_count_ = reinterpret_cast<int *>(AllocateMem(sizeof(int)));
  action_ = reinterpret_cast<int *>(AllocateMem(sizeof(int)));
  prior_ = reinterpret_cast<float *>(AllocateMem(sizeof(float)));
  // Set init value
  Memset(total_reward_, 0, sizeof(float));
  Memset(explore_count_, 0, sizeof(int));
  if (action != nullptr) {
    Memcpy(action_, action, sizeof(int));
  } else {
    action_ = nullptr;
  }
  if (prior != nullptr) {
    Memcpy(prior_, prior, sizeof(float));
  } else {
    prior_ = nullptr;
  }
}

int CPUMonteCarloTreeNode::GetMaxPosition(float *selection_value, int num_items,
                                          void *device_stream) {
  int max_index = -1;
  float max_value = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < num_items; i++) {
    if (selection_value[i] > max_value) {
      max_value = selection_value[i];
      max_index = i;
    }
  }
  return max_index;
}

MonteCarloTreeNodePtr CPUMonteCarloTreeNode::BestAction() const {
  return *std::max_element(children_.begin(), children_.end(),
                           [](const MonteCarloTreeNodePtr node_a,
                              const MonteCarloTreeNodePtr node_b) {
                             return node_a->BestActionPolicy(node_b);
                           });
}

bool CPUMonteCarloTreeNode::BestActionPolicy(MonteCarloTreeNodePtr node) const {
  float outcome_self = (outcome_.empty() ? 0 : outcome_[player_]);
  float outcome_input =
      (node->outcome().empty() ? 0 : node->outcome()[node->player()]);
  if (outcome_self != outcome_input) {
    return outcome_self < outcome_input;
  }
  if (*explore_count_ != *(node->explore_count())) {
    return *explore_count_ < *(node->explore_count());
  }
  return *total_reward_ < *(node->total_reward());
}

void *CPUMonteCarloTreeNode::AllocateMem(size_t size) { return malloc(size); }

bool CPUMonteCarloTreeNode::Memcpy(void *dst_ptr, void *src_ptr, size_t size) {
  std::memcpy(dst_ptr, src_ptr, size);
  return true;
}

bool CPUMonteCarloTreeNode::MemcpyAsync(void *dst_ptr, void *src_ptr,
                                        size_t size, void *device_stream) {
  std::memcpy(dst_ptr, src_ptr, size);
  return true;
}

bool CPUMonteCarloTreeNode::Memset(void *dst_ptr, int value, size_t size) {
  memset(dst_ptr, value, size);
  return true;
}

bool CPUMonteCarloTreeNode::Free(void *ptr) {
  free(ptr);
  return true;
}
} // namespace utils
} // namespace mindspore_rl
