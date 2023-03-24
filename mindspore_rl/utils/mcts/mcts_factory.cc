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

#include <utils/mcts/mcts_factory.h>
#include <iostream>
#include "utils/mcts/logging.h"

MonteCarloTreeFactory& MonteCarloTreeFactory::GetInstance() {
  static MonteCarloTreeFactory instance;
  return instance;
}

MonteCarloTreeNodePtr MonteCarloTreeFactory::CreateNode(const std::string &node_name, int *action, float *prior,
                                                        float *init_reward, int player, int64_t tree_handle,
                                                        MonteCarloTreeNodePtr parent_node, int row, int state_size) {
  auto node_creator = map_node_name_to_node_creator_.find(node_name);
  if (node_creator == map_node_name_to_node_creator_.end()) {
    std::ostringstream oss;
    oss << "The input node name " << node_name
        << " in CreateNode does not exist. ";
    oss << "Node register: [";
    for (auto iter = map_node_name_to_node_creator_.begin(); iter != map_node_name_to_node_creator_.end(); iter++) {
      oss << iter->first << " ";
    }
    oss << "]";
    LOG_ERROR("%s", oss.str().c_str());
    // Return nullptr to catch the exception outside.
    return nullptr;
  }
  auto node = std::shared_ptr<MonteCarloTreeNode>(
      node_creator->second(node_name, action, prior, init_reward, player, tree_handle, parent_node, row, state_size));
  return node;
}

std::tuple<int64_t, MonteCarloTreePtr> MonteCarloTreeFactory::CreateTree(const std::string &tree_name,
                                                                         const std::string &node_name, int player,
                                                                         float max_utility, int state_size,
                                                                         int total_num_player,
                                                                         float *input_global_variable) {
  handle_++;
  MonteCarloTreePtr tree;
  auto root = MonteCarloTreeFactory::GetInstance().CreateNode(node_name, nullptr, nullptr, nullptr, player, handle_,
                                                              nullptr, 0, state_size);
  if (root == nullptr) {
    return std::make_tuple(handle_, nullptr);
  }

  auto tree_creator = map_tree_name_to_tree_creator_.find(tree_name);
  if (tree_creator == map_tree_name_to_tree_creator_.end()) {
    std::ostringstream oss;
    oss << "The input tree name " << tree_name
        << " in CreateTree does not exist. ";
    oss << "Tree register: [";
    for (auto iter = map_tree_name_to_tree_creator_.begin(); iter != map_tree_name_to_tree_creator_.end(); iter++) {
      oss << iter->first << " ";
    }
    oss << "]";
    LOG_ERROR("%s", oss.str().c_str());
    // Return nullptr to catch the exception outside.
    return std::make_tuple(handle_, nullptr);
    }
    tree =
      std::shared_ptr<MonteCarloTree>(tree_creator->second(root, max_utility, handle_, state_size, total_num_player));

    map_handle_to_tree_ptr_.insert(std::make_pair(handle_, tree));
    map_handle_to_tree_const_.insert(std::make_pair(handle_, input_global_variable));
    return std::make_tuple(handle_, tree);
}

void MonteCarloTreeFactory::InsertGlobalVariable(int64_t tree_handle, float *global_variable) {
    map_handle_to_tree_variable_.insert(
        std::make_pair(tree_handle, global_variable));
}

void MonteCarloTreeFactory::RegisterNode(const std::string& node_name, NodeCreator&& node_creator) {
  map_node_name_to_node_creator_.insert(std::make_pair(node_name, node_creator));
}

void MonteCarloTreeFactory::RegisterTree(const std::string& tree_name, TreeCreator&& tree_creator) {
  map_tree_name_to_tree_creator_.insert(std::make_pair(tree_name, tree_creator));
}

MonteCarloTreePtr MonteCarloTreeFactory::GetTreeByHandle(int64_t handle) {
  auto iter = map_handle_to_tree_ptr_.find(handle);
  if (iter == map_handle_to_tree_ptr_.end()) {
    std::ostringstream oss;
    oss << "The input handle " << handle
        << " in GetTreeByHandle does not exist. ";
    oss << "Handle register: [";
    for (auto pairs = map_handle_to_tree_ptr_.begin(); pairs != map_handle_to_tree_ptr_.end(); pairs++) {
      oss << pairs->first << " ";
    }
    oss << "]";
    LOG_ERROR("%s", oss.str().c_str());
    // Return nullptr to catch the exception outside.
    return nullptr;
  }
  return iter->second;
}

float *MonteCarloTreeFactory::GetTreeVariableByHandle(int64_t handle) {
  auto iter = map_handle_to_tree_variable_.find(handle);
  if (iter == map_handle_to_tree_variable_.end()) {
    std::ostringstream oss;
    oss << "The input handle " << handle
        << " in GetTreeVariableByHandle does not exist. ";
    oss << "Handle register: [";
    for (auto pairs = map_handle_to_tree_variable_.begin(); pairs != map_handle_to_tree_variable_.end(); pairs++) {
      oss << pairs->first << " ";
    }
    oss << "]";
    LOG_ERROR("%s", oss.str().c_str());
    // Return nullptr to catch the exception outside.
    float *null_vector = nullptr;
    return null_vector;
  }
  return iter->second;
}

float *MonteCarloTreeFactory::GetTreeConstByHandle(int64_t handle) {
  auto iter = map_handle_to_tree_const_.find(handle);
  if (iter == map_handle_to_tree_const_.end()) {
    std::ostringstream oss;
    oss << "The input handle " << handle
        << " in GetTreeVariableByHandle does not exist. ";
    oss << "Handle register: [";
    for (auto pairs = map_handle_to_tree_const_.begin(); pairs != map_handle_to_tree_const_.end(); pairs++) {
      oss << pairs->first << " ";
    }
    oss << "]";
    LOG_ERROR("%s", oss.str().c_str());
    // Return nullptr to catch the exception outside.
    float *null_vector = nullptr;
    return null_vector;
  }
  return iter->second;
}

bool MonteCarloTreeFactory::DeleteTree(int64_t handle) {
  auto iter = map_handle_to_tree_ptr_.find(handle);
  if (iter == map_handle_to_tree_ptr_.end()) {
    std::ostringstream oss;
    oss << "The input handle " << handle << " in DeleteTree does not exist. ";
    oss << "Handle register: [";
    for (auto pairs = map_handle_to_tree_ptr_.begin(); pairs != map_handle_to_tree_ptr_.end(); pairs++) {
      oss << pairs->first << " ";
    }
    oss << "]";
    LOG_ERROR("%s", oss.str().c_str());
    return false;
  } else {
    map_handle_to_tree_ptr_.erase(handle);
    return true;
  }
}

bool MonteCarloTreeFactory::DeleteTreeVariable(int64_t handle) {
  auto iter = map_handle_to_tree_variable_.find(handle);
  if (iter == map_handle_to_tree_variable_.end()) {
    std::ostringstream oss;
    oss << "The input handle " << handle
        << " in DeleteTreeVariable does not exist. ";
    oss << "Handle register: [";
    for (auto pairs = map_handle_to_tree_variable_.begin(); pairs != map_handle_to_tree_variable_.end(); pairs++) {
      oss << pairs->first << " ";
    }
    oss << "]";
    LOG_ERROR("%s", oss.str().c_str());
    return false;
  } else {
    map_handle_to_tree_variable_.erase(handle);
    return true;
  }
}
