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

#include <utils/mcts/cpu/cpu_muzero.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <numeric>
namespace mindspore_rl {
namespace utils {
using namespace std;

bool CPUMuzeroTreeNode::SelectionPolicy(float *uct_value,
                                        void *device_stream) const {
  float value_score = 0;
  if (!outcome_.empty()) {
    *uct_value = outcome_[player_];
    return true;
  }
  auto global_const_vector =
      MonteCarloTreeFactory::GetInstance().GetTreeConstByHandle(tree_handle_);
  auto global_variable_vector =
      MonteCarloTreeFactory::GetInstance().GetTreeVariableByHandle(
          tree_handle_);
  float minimum = global_variable_vector[0];
  float maximum = global_variable_vector[1];
  float pb_c_base = global_const_vector[1];
  float pb_c_init = global_const_vector[2];

  float pb_c =
      log((*parent_->explore_count() + pb_c_base + 1) / pb_c_base) + pb_c_init;
  pb_c *= sqrt(*parent_->explore_count()) / (*explore_count_ + 1);
  float prior_score = pb_c * *prior_;
  if (*explore_count_ != 0) {
    float no_norm_value_score = *total_reward_ / *explore_count_;
    if (maximum > minimum) {
      value_score = (no_norm_value_score - minimum) / (maximum - minimum);
    }
  }
  *uct_value = prior_score + value_score;
  return true;
}

bool CPUMuzeroTreeNode::Update(float *values, int total_num_player,
                               void *device_stream) {
  *total_reward_ += values[player_];
  *explore_count_ += 1;
  auto global_variable_vector =
      MonteCarloTreeFactory::GetInstance().GetTreeVariableByHandle(
          tree_handle_);
  auto global_const_vector =
      MonteCarloTreeFactory::GetInstance().GetTreeConstByHandle(tree_handle_);
  float minimum = global_variable_vector[0];
  float maximum = global_variable_vector[1];
  float discount = global_const_vector[0];
  float no_norm_value_score = *total_reward_ / *explore_count_;
  global_variable_vector[0] = min(minimum, no_norm_value_score);
  global_variable_vector[1] = max(maximum, no_norm_value_score);
  *values = *node_reward_ + discount * *values;
  return true;
}

void CPUMuzeroTreeNode::SetInitReward(float *init_reward) {
  if (action_ == nullptr) {
    init_reward[0] = 0;
  }
  memcpy(node_reward_, init_reward, sizeof(float));
}

MonteCarloTreeNodePtr CPUMuzeroTreeNode::BestAction() const {
  auto global_variable_vector =
      MonteCarloTreeFactory::GetInstance().GetTreeVariableByHandle(
          tree_handle_);
  float temperature = global_variable_vector[2];

  random_device rd;
  mt19937 gen(rd());

  vector<float> explore_count_vector;
  vector<float> accumulated_count;
  for (auto &child : children_) {
    auto prob = pow(*child->explore_count(), (1 / temperature));
    explore_count_vector.emplace_back(prob);
  }
  int sum_value =
      accumulate(explore_count_vector.begin(), explore_count_vector.end(), 0);
  uniform_int_distribution<int> dis(1, sum_value);
  partial_sum(explore_count_vector.begin(), explore_count_vector.end(),
              back_inserter(accumulated_count));
  int pos = dis(gen);
  int index =
      lower_bound(accumulated_count.begin(), accumulated_count.end(), pos) -
      accumulated_count.begin();

  return children_[index];
}
} // namespace utils
} // namespace mindspore_rl
