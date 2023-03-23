/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_RL_UTILS_MCTS_CPU_CPU_MUZERO_H_
#define MINDSPORE_RL_UTILS_MCTS_CPU_CPU_MUZERO_H_

#include <utils/mcts/mcts_factory.h>
#include <utils/mcts/cpu/cpu_mcts_tree_node.h>
#include <string>

class CPUMuzeroTreeNode : public CPUMonteCarloTreeNode {
 public:
  CPUMuzeroTreeNode(const std::string &name, int *action, float *prior, float *init_reward, int player,
                    int64_t tree_handle, MonteCarloTreeNodePtr parent_node, int row, int state_size)
      : CPUMonteCarloTreeNode(name, action, prior, init_reward, player, tree_handle, parent_node, row, state_size) {
    node_reward_ = new float[sizeof(float)];
    memset(node_reward_, 0, sizeof(float));
  }

  bool SelectionPolicy(float *uct_value, void *device_stream) const override;
  bool Update(float *value, int total_num_player, void *device_stream) override;
  void SetInitReward(float *init_reward) override;
  MonteCarloTreeNodePtr BestAction() const override;

 private:
  float *node_reward_;
};
MS_REG_NODE(CPUMuzero, CPUMuzeroTreeNode);

#endif  // MINDSPORE_RL_UTILS_MCTS_CPU_CPU_MUZERO_H_
