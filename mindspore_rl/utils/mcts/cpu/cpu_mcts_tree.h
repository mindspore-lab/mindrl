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

#ifndef MINDSPORE_RL_UTILS_MCTS_CPU_CPU_MCTS_TREE_H_
#define MINDSPORE_RL_UTILS_MCTS_CPU_CPU_MCTS_TREE_H_

#include <utils/mcts/mcts_tree.h>
#include <utils/mcts/mcts_factory.h>
#include <cstring>
#include <string>

class CPUMonteCarloTree : public MonteCarloTree {
 public:
  CPUMonteCarloTree(MonteCarloTreeNodePtr root, float max_utility, int64_t tree_handle, int state_size,
                    int total_num_player)
      : MonteCarloTree(root, max_utility, tree_handle, state_size, total_num_player) {}

  ~CPUMonteCarloTree() override = default;

  bool Expansion(std::string node_name, int *action, float *prior, float *init_reward, int num_action,
                 int state_size) override;

  void *AllocateMem(size_t size) override;
  bool Memcpy(void *dst_ptr, void *src_ptr, size_t size) override;
  bool Memset(void *dst_ptr, int value, size_t size) override;
  bool Free(void *ptr) override;
};

MS_REG_TREE(CPUCommon, CPUMonteCarloTree);

#endif  // MINDSPORE_RL_UTILS_MCTS_CPU_CPU_MCTS_TREE_H_
