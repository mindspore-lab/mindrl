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

#include <utils/mcts/gpu/gpu_vanilla.h>
#include <utils/mcts/gpu/cuda_impl/vanilla_impl.cuh>
#include <cmath>
#include <iostream>
#include <limits>
namespace mindspore_rl {
namespace utils {
bool GPUVanillaTreeNode::SelectionPolicy(float *uct_value,
                                         void *device_stream) const {
  if (!outcome_.empty()) {
    auto out_value = outcome_[player_];
    cudaMemcpy(uct_value, &out_value, sizeof(float), cudaMemcpyHostToDevice);
    return true;
  }

  auto global_variable_vector =
      MonteCarloTreeFactory::GetInstance().GetTreeConstByHandle(tree_handle_);
  auto uct_ptr = global_variable_vector;

  int *parent_explore_count = parent_->explore_count();
  CalSelectionPolicy(explore_count_, total_reward_, parent_explore_count,
                     uct_ptr, uct_value,
                     static_cast<cudaStream_t>(device_stream));
  return true;
}

bool GPUVanillaTreeNode::Update(float *values, int total_num_player,
                                void *device_stream) {
  CalUpdate(explore_count_, total_reward_, values, player_,
            reinterpret_cast<cudaStream_t>(device_stream));

  return true;
}
} // namespace utils
} // namespace mindspore_rl
