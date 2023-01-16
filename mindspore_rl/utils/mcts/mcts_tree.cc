/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "utils/mcts/mcts_tree.h"
#include <algorithm>
#include "utils/mcts/mcts_factory.h"
#include "utils/mcts/mcts_tree_node.h"

bool MonteCarloTree::Selection(int *action_list, int max_action, void *device_stream) {
  visited_path_.clear();
  visited_path_.emplace_back(root_);
  MonteCarloTreeNodePtr current_node = root_;
  // Create a max length action to avoid dynamic shape
  int i = 0;
  MonteCarloTreeNodePtr selected_child = nullptr;
  while (!current_node->IsLeafNode()) {
    selected_child = current_node->SelectChild(device_stream);
    if (selected_child == nullptr) {
      return false;
    }
    if (max_action != -1) {
      Memcpy(action_list + i, selected_child->action(), sizeof(int));
      i++;
    }
    visited_path_.emplace_back(selected_child);
    current_node = selected_child;
  }
  // If max_action is -1, which means that the Selection will only return the last action.
  if (max_action == -1 && selected_child != nullptr) {
    Memcpy(action_list, selected_child->action(), sizeof(int));
  }
  placeholder_handle_++;
  return true;
}

bool MonteCarloTree::Backpropagation(float *returns, void *device_stream) {
  // Reverse the visited path, update from the bottom to the top.
  auto leaf_node = visited_path_[visited_path_.size() - 1];
  bool solved = false;
  // If the leaf node is terminal, which means that this branch is solved.
  if (leaf_node->terminal()) {
    solved = true;
  }
  // For each node in visited path, call the Update() to update the value.
  // If current branch is solved, backprop the best outcome from the bottom to top.
  // for (auto &node : visited_path_) {
  for (int i = visited_path_.size() - 1; i >= 0; i--) {
    auto node = visited_path_[i];
    node->Update(returns, total_num_player_, device_stream);
    if (solved && !node->IsLeafNode()) {
      MonteCarloTreeNodePtr best = nullptr;
      bool all_solved = true;
      for (const auto &child : node->children()) {
        if (child->outcome().empty()) {
          all_solved = false;
        } else if (best == nullptr || child->outcome()[child->player()] > best->outcome()[best->player()]) {
          best = child;
        }
      }
      if (best != nullptr && (all_solved || best->outcome()[best->player()] == max_utility_)) {
        node->set_outcome(best->outcome());
      } else {
        solved = false;
      }
    }
  }
  bool fully_solved = !root_->outcome().empty();
  return fully_solved;
}

int *MonteCarloTree::BestAction() {
  auto best_child_node = root_->BestAction();
  return best_child_node->action();
}
