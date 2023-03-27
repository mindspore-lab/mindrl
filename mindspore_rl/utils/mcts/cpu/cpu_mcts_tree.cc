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

#include <utils/mcts/cpu/cpu_mcts_tree.h>
namespace mindspore_rl {
namespace utils {
bool CPUMonteCarloTree::Expansion(std::string node_name, int *action,
                                  float *prior, float *init_reward,
                                  int num_action, int state_size) {
  // Expand the last node of visited_path.
  auto leaf_node = visited_path_.at(visited_path_.size() - 1);
  if (init_reward != nullptr) {
    leaf_node->SetInitReward(init_reward);
  }
  int player = (leaf_node->action() == nullptr)
                   ? (leaf_node->player())
                   : ((leaf_node->player() + 1) % total_num_player_);
  for (int i = 0; i < num_action; i++) {
    if (*(action + i) != -1) {
      auto child_node = MonteCarloTreeFactory::GetInstance().CreateNode(
          node_name, action + i, prior + i, init_reward, player, tree_handle_,
          leaf_node, leaf_node->row() + 1, state_size);
      leaf_node->AddChild(child_node);
    }
  }
  return true;
}

void *CPUMonteCarloTree::AllocateMem(size_t size) { return malloc(size); }

bool CPUMonteCarloTree::Memcpy(void *dst_ptr, void *src_ptr, size_t size) {
  std::memcpy(dst_ptr, src_ptr, size);
  return true;
}

bool CPUMonteCarloTree::Memset(void *dst_ptr, int value, size_t size) {
  memset(dst_ptr, value, size);
  return true;
}

bool CPUMonteCarloTree::Free(void *ptr) {
  free(ptr);
  return true;
}
} // namespace utils
} // namespace mindspore_rl
