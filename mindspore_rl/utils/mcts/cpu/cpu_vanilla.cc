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

#include <utils/mcts/cpu/cpu_vanilla.h>
#include <cmath>
#include <iostream>
#include <limits>
namespace mindspore_rl {
namespace utils {
bool CPUVanillaTreeNode::SelectionPolicy(float *uct_value,
                                         void *device_stream) const {
  if (!outcome_.empty()) {
    *uct_value = outcome_[player_];
    return true;
  }
  if (*explore_count_ == 0) {
    *uct_value = std::numeric_limits<float>::infinity();
    return true;
  }

  auto global_variable_vector =
      MonteCarloTreeFactory::GetInstance().GetTreeConstByHandle(tree_handle_);
  if (global_variable_vector == nullptr) {
    return false;
  }
  auto uct_ptr = global_variable_vector[0];
  *uct_value = *total_reward_ / *explore_count_ +
               uct_ptr * std::sqrt(std::log(*(parent_->explore_count())) /
                                   *explore_count_);
  return true;
}

bool CPUVanillaTreeNode::Update(float *values, int total_num_player,
                                void *device_stream) {
  *total_reward_ += values[player_];
  *explore_count_ += 1;
  return true;
}
} // namespace utils
} // namespace mindspore_rl
