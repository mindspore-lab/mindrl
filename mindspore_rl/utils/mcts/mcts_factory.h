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

#ifndef MINDSPORE_RL_UTILS_MCTS_MCTS_FACTORY_H_
#define MINDSPORE_RL_UTILS_MCTS_MCTS_FACTORY_H_

#include <utils/mcts/mcts_tree.h>
#include <utils/mcts/mcts_tree_node.h>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

constexpr int64_t kInvalidHandle = -1;
using NodeCreator = std::function<MonteCarloTreeNode *(std::string, int *, float *, float *, int, int64_t,
                                                       MonteCarloTreeNodePtr, int, int)>;
using TreeCreator = std::function<MonteCarloTree *(MonteCarloTreeNodePtr, float, int64_t, int, int)>;

class MonteCarloTreeFactory {
 public:
  // Create a factory instance of MonteCarloTree.
  static MonteCarloTreeFactory& GetInstance();
  // Create a subclass of MonteCarloTreeNode based on input node_name.
  // It will return the pointer of this instance.
  MonteCarloTreeNodePtr CreateNode(const std::string &node_name, int *action, float *prior, float *init_reward,
                                   int player, int64_t tree_handle, MonteCarloTreeNodePtr parent_node, int row,
                                   int state_size);
  // Create a MonteCarloTree based on input tree_name.
  // It will return the unique handle of this tree and its pointer.
  std::tuple<int64_t, MonteCarloTreePtr> CreateTree(const std::string &tree_name, const std::string &node_name,
                                                    int player, float max_utility, int state_size, int total_num_player,
                                                    float *input_global_variable);
  void InsertGlobalVariable(int64_t tree_handle, float *global_variable);
  // Insert the node_creator to a map (key: node_name, value: node_creator).
  void RegisterNode(const std::string &node_name, NodeCreator &&node_creator);
  // Insert the tree_creator to a map (key: tree_name, value: tree_creator).
  void RegisterTree(const std::string& tree_name, TreeCreator&& tree_creator);
  // Get the tree instance by the unique handle.
  MonteCarloTreePtr GetTreeByHandle(int64_t handle);
  // Get the global variable by the unique handle.
  float *GetTreeVariableByHandle(int64_t handle);
  float *GetTreeConstByHandle(int64_t handle);

  // Erase the tree and all the nodes which matches the input handle.
  bool DeleteTree(int64_t handle);
  // Erase the tree and all the nodes which matches the input handle.
  bool DeleteTreeVariable(int64_t handle);

 private:
  MonteCarloTreeFactory() = default;
  ~MonteCarloTreeFactory() = default;

  std::map<std::string, NodeCreator> map_node_name_to_node_creator_;
  std::map<std::string, TreeCreator> map_tree_name_to_tree_creator_;
  std::map<int64_t, MonteCarloTreePtr> map_handle_to_tree_ptr_;
  std::map<int64_t, float *> map_handle_to_tree_variable_;
  std::map<int64_t, float *> map_handle_to_tree_const_;
  int64_t handle_ = kInvalidHandle;
};

class MonteCarloTreeNodeRegister {
 public:
  MonteCarloTreeNodeRegister(const std::string& node_name, NodeCreator&& node_creator) {
    MonteCarloTreeFactory::GetInstance().RegisterNode(node_name, std::move(node_creator));
  }
};

class MonteCarloTreeRegister {
 public:
  MonteCarloTreeRegister(const std::string& tree_name, TreeCreator&& tree_creator) {
    MonteCarloTreeFactory::GetInstance().RegisterTree(tree_name, std::move(tree_creator));
  }
};

// Helper registration macro for NODECLASS
// When user inherits the base class of MonteCarloTreeNode, user can register the class by NAME.
// Then user can pass the NAME in python side to create derived class in C++ side.
#define MS_REG_NODE(NAME, NODECLASS)                                                                           \
  static_assert(std::is_base_of<MonteCarloTreeNode, NODECLASS>::value, " must be base of MonteCarloTreeNode"); \
  static const MonteCarloTreeNodeRegister montecarlo_##NAME##_node_reg(                                        \
    #NAME, [](std::string name, int *action, float *prior, float *reward, int player, int64_t tree_handle,     \
              MonteCarloTreeNodePtr parent_node, int row, int state_size) {                                    \
      return new NODECLASS(name, action, prior, reward, player, tree_handle, parent_node, row, state_size);    \
    });

// Helper registration macro for TREECLASS
// When user inherits the base class of MonteCarloTree, user can register the class by NAME.
// Then user can pass the NAME in python side to create derived class in C++ side.
#define MS_REG_TREE(NAME, TREECLASS)                                                                               \
  static_assert(std::is_base_of<MonteCarloTree, TREECLASS>::value, " must be base of MonteCarloTree");             \
  static const MonteCarloTreeRegister montecarlo_##NAME##_tree_reg(                                                \
    #NAME,                                                                                                         \
    [](MonteCarloTreeNodePtr root, float max_utility, int64_t tree_handle, int state_size, int total_num_player) { \
      return new TREECLASS(root, max_utility, tree_handle, state_size, total_num_player);                          \
    });

#endif  // MINDSPORE_RL_UTILS_MCTS_MCTS_FACTORY_H_
