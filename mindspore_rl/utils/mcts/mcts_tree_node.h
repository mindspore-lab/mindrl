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

#ifndef MINDSPORE_RL_UTILS_MCTS_MCTS_TREE_NODE_H_
#define MINDSPORE_RL_UTILS_MCTS_MCTS_TREE_NODE_H_

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <cstring>
namespace mindspore_rl {
namespace utils {
class MonteCarloTreeNode {
public:
  // The base class of MonteCarloTreeNode.
  MonteCarloTreeNode(std::string name, int *action, float *prior,
                     float *init_reward, int player, int64_t tree_handle,
                     std::shared_ptr<MonteCarloTreeNode> parent_node, int row,
                     int state_size)
      : name_(name), player_(player), row_(row), terminal_(false),
        tree_handle_(tree_handle), parent_(parent_node) {}

  virtual ~MonteCarloTreeNode() = default;

  // Init node function
  virtual void InitNode(int state_size, float *init_reward, int *action,
                        float *prior) = 0;

  // It will select the child whose value of SelectionPolicy is the highest.
  std::shared_ptr<MonteCarloTreeNode> SelectChild(void *device_stream);

  // The virtual function of SelectionPolicy. In this function, user needs to
  // implement the rule to select child node, such as UCT(UCB) function, RAVE,
  // AMAF, etc.
  virtual bool SelectionPolicy(float *uct_value, void *device_stream) const = 0;
  virtual int GetMaxPosition(float *selection_value, int num_items,
                             void *device_stream) = 0;

  // The virtual function of Update. It is invoked during backpropagation. User
  // needs implement how to update the local value according to the input
  // returns.
  virtual bool Update(float *returns, int total_num_player,
                      void *device_stream) = 0;
  virtual void SetInitReward(float *reward) = 0;

  // After the whole tree finished, use BestAction to obtain the best action for
  // the root.
  virtual std::shared_ptr<MonteCarloTreeNode> BestAction() const = 0;

  // The policy to choose BestAction
  // The default policy is that:
  // 1. First compare the outcome of two nodes
  // 2. If both of them does not have outcome (or same), then compare the
  // explore_count_
  // 3. If they have the same explore_count_, then compare the total_reward_
  virtual bool
  BestActionPolicy(std::shared_ptr<MonteCarloTreeNode> child_node) const = 0;

  bool IsLeafNode() { return children_.empty(); }
  bool AddChild(std::shared_ptr<MonteCarloTreeNode> child) {
    if (child == nullptr) {
      return false;
    }
    children_.emplace_back(child);
    return true;
  }
  std::vector<std::shared_ptr<MonteCarloTreeNode>> children() {
    return children_;
  }

  virtual void *AllocateMem(size_t size) = 0;
  virtual bool Memcpy(void *dst_ptr, void *src_ptr, size_t size) = 0;
  virtual bool MemcpyAsync(void *dst_ptr, void *src_ptr, size_t size,
                           void *device_stream) = 0;
  virtual bool Memset(void *dst_ptr, int value, size_t size) = 0;
  virtual bool Free(void *ptr) = 0;

  void FreeNode() {
    for (auto &child : children_) {
      child->FreeNode();
    }
    Free(state_);
    Free(total_reward_);
    Free(explore_count_);
    Free(action_);
    Free(prior_);
    children_.clear();
    outcome_.clear();
  }

  void set_state(float *input_state, int state_size) {
    Memcpy(state_, input_state, sizeof(float) * state_size);
  }
  float *state() { return state_; }

  void set_terminal(bool done) { terminal_ = done; }
  bool terminal() { return terminal_; }

  void set_outcome(std::vector<float> new_outcome) { outcome_ = new_outcome; }
  std::vector<float> outcome() { return outcome_; }

  int *action() { return action_; }
  int row() { return row_; }
  int player() { return player_; }
  int *explore_count() { return explore_count_; }
  float *total_reward() { return total_reward_; }

  virtual std::string DebugString() = 0;

protected:
  std::string name_; // The name of this node.
  bool terminal_;    // Whether current node is terminal node.
  int row_;          // Which row this node belongs to (for DEBUG).
  int *action_; // The action that transfers from parent node to current node.
  float
      *prior_; // P(a|s), the probability that choose this node in parent node.
  float *state_;        // The state current node states for.
  int player_;          // This node belongs to which player.
  int *explore_count_;  // Number of times that current node is visited.
  float *total_reward_; // The total reward of current node.
  int64_t tree_handle_; // Current node belongs to which tree.

  std::vector<float> outcome_;                 // The outcome of terminal node.
  std::shared_ptr<MonteCarloTreeNode> parent_; // Parent node.
  std::vector<std::shared_ptr<MonteCarloTreeNode>> children_; // All child node.
};
using MonteCarloTreeNodePtr = std::shared_ptr<MonteCarloTreeNode>;
} // namespace utils
} // namespace mindspore_rl
#endif // MINDSPORE_RL_UTILS_MCTS_MCTS_TREE_NODE_H_
