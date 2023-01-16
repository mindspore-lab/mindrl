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

#ifndef MINDSPORE_RL_UTILS_MCTS_GPU_GPU_MCTS_TREE_NODE_H_
#define MINDSPORE_RL_UTILS_MCTS_GPU_GPU_MCTS_TREE_NODE_H_

#include <utils/mcts/mcts_tree_node.h>
#include <utils/mcts/gpu/cuda_impl/argmax_impl.cuh>
#include <cuda_runtime_api.h>
#include <vector>
#include <algorithm>
#include <memory>
#include <string>
#include <cstring>

class GPUMonteCarloTreeNode : public MonteCarloTreeNode {
 public:
  GPUMonteCarloTreeNode(std::string name, int *action, float *prior, float *init_reward, int player,
                        int64_t tree_handle, std::shared_ptr<MonteCarloTreeNode> parent_node, int row, int state_size)
      : MonteCarloTreeNode(name, action, prior, init_reward, player, tree_handle, parent_node, row, state_size) {
    if (state_size > 0) {
      InitNode(state_size, init_reward, action, prior);
    } else {
      std::cout << "[ERROR]The state size is smaller than 0, please check" << std::endl;
    }
  }

  ~GPUMonteCarloTreeNode() override = default;

  void InitNode(int state_size, float *init_reward, int *action, float *prior) override;
  int GetMaxPosition(float *selection_value, int num_items, void *device_stream) override;
  bool BestActionPolicy(std::shared_ptr<MonteCarloTreeNode> child_node) const override;
  virtual void SetInitReward(float *init_reward) { Memcpy(total_reward_, init_reward + player_, sizeof(float)); }
  std::shared_ptr<MonteCarloTreeNode> BestAction() const override;

  std::string DebugString() override {
    int *action_host = new int[sizeof(int)];
    if (action_ != nullptr) {
      cudaMemcpy(action_host, action_, sizeof(int), cudaMemcpyDeviceToHost);
    } else {
      *action_host = -1;
    }
    std::ostringstream oss;
    oss << tree_handle_ << "_" << name_ << "_row_" << row_ << "_player_" << player_;
    oss << "_action_" << *action_host << "_terminal_" << terminal_;
    return oss.str();
  }

  void *AllocateMem(size_t size) override;
  bool Memcpy(void *dst_ptr, void *src_ptr, size_t size) override;
  bool MemcpyAsync(void *dst_ptr, void *src_ptr, size_t size, void *device_stream) override;
  bool Memset(void *dst_ptr, int value, size_t size) override;
  bool Free(void *ptr) override;

  virtual bool SelectionPolicy(float *uct_value, void *device_stream) const = 0;
  virtual bool Update(float *returns, int num_player, void *device_stream) = 0;
};

#endif  // MINDSPORE_RL_UTILS_MCTS_GPU_GPU_MCTS_TREE_NODE_H_
