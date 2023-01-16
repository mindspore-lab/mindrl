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

#include <utils/mcts/gpu/gpu_mcts_tree_node.h>
#include <utils/mcts/gpu/cuda_impl/argmax_impl.cuh>
#include <cuda_runtime_api.h>
#include <limits>

void GPUMonteCarloTreeNode::InitNode(int state_size, float *init_reward, int *action, float *prior) {
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

int GPUMonteCarloTreeNode::GetMaxPosition(float *selection_value, int num_items, void *device_stream) {
  int *index = reinterpret_cast<int *>(AllocateMem(sizeof(int)));
  CalArgmax(selection_value, num_items, index, static_cast<cudaStream_t>(device_stream));
  int *index_host = new int[sizeof(int)];
  cudaMemcpyAsync(index_host, index, sizeof(int), cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(device_stream));
  cudaStreamSynchronize(static_cast<cudaStream_t>(device_stream));

  return *index_host;
}

MonteCarloTreeNodePtr GPUMonteCarloTreeNode::BestAction() const {
  return *std::max_element(children_.begin(), children_.end(),
                           [](const MonteCarloTreeNodePtr node_a, const MonteCarloTreeNodePtr node_b) {
                             return node_a->BestActionPolicy(node_b);
                           });
}

bool GPUMonteCarloTreeNode::BestActionPolicy(MonteCarloTreeNodePtr node) const {
  int *explore_count_host = reinterpret_cast<int *>(malloc(sizeof(int)));
  float *total_reward_host = reinterpret_cast<float *>(malloc(sizeof(float)));
  int *explore_count_input_host = reinterpret_cast<int *>(malloc(sizeof(int)));
  float *total_reward_input_host = reinterpret_cast<float *>(malloc(sizeof(float)));

  cudaMemcpy(explore_count_host, explore_count_, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(total_reward_host, total_reward_, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(explore_count_input_host, node->explore_count(), sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(total_reward_input_host, node->total_reward(), sizeof(float), cudaMemcpyDeviceToHost);

  float outcome_self = (outcome_.empty() ? 0 : outcome_[player_]);
  float outcome_input = (node->outcome().empty() ? 0 : node->outcome()[node->player()]);
  if (outcome_self != outcome_input) {
    return outcome_self < outcome_input;
  }
  if (*explore_count_host != *(explore_count_input_host)) {
    return *explore_count_host < *(explore_count_input_host);
  }
  return *total_reward_host < *(total_reward_input_host);
}

void *GPUMonteCarloTreeNode::AllocateMem(size_t size) {
  void *device_state_ptr = nullptr;
  cudaMalloc(&device_state_ptr, size);
  return device_state_ptr;
}

bool GPUMonteCarloTreeNode::Memcpy(void *dst_ptr, void *src_ptr, size_t size) {
  cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDeviceToDevice);
  return true;
}

bool GPUMonteCarloTreeNode::MemcpyAsync(void *dst_ptr, void *src_ptr, size_t size, void *device_stream) {
  cudaMemcpyAsync(dst_ptr, src_ptr, size, cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(device_stream));
  return true;
}

bool GPUMonteCarloTreeNode::Memset(void *dst_ptr, int value, size_t size) {
  cudaMemset(dst_ptr, value, size);
  return true;
}

bool GPUMonteCarloTreeNode::Free(void *ptr) {
  cudaFree(ptr);
  return true;
}
