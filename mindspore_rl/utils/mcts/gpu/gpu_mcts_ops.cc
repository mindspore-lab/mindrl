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

#include <utils/mcts/mcts_factory.h>
#include <utils/mcts/mcts_tree.h>
#include <utils/mcts/mcts_tree_node.h>
#include <utils/mcts/custom_aot_extra.h>
#include <cuda_runtime_api.h>
#include <cstdint>
#include <iostream>
namespace mindspore_rl {
namespace utils {
constexpr int kErrorCode = 2;
constexpr int kInputIndex = 1;

class CreationAttr : public AotKernelData {
public:
  std::string tree_type;
  std::string node_type;
  float max_utility;
  float state_size;
  float player;
  float total_num_player;
};

extern "C" int MctsCreationInit(int *ndims, int64_t **shapes,
                                const char **dtypes, AotExtra *extra) {
  CreationAttr *kernel_ptr = new CreationAttr;
  kernel_ptr->tree_type = extra->Attr<std::string>("tree_type");
  kernel_ptr->node_type = extra->Attr<std::string>("node_type");
  kernel_ptr->max_utility = extra->Attr<float>("max_utility");
  kernel_ptr->state_size = extra->Attr<float>("state_size");
  kernel_ptr->player = extra->Attr<float>("player");
  kernel_ptr->total_num_player = extra->Attr<float>("total_num_player");
  extra->SetKernelData(kernel_ptr);
  return 0;
}

extern "C" int MctsCreation(int nparam, void **params, int *ndims,
                            int64_t **shapes, const char **dtypes, void *stream,
                            void *extra) {
  // Input Attr
  // MctsCreation has 6 compulsory attr value
  // 1. The name of tree_type
  // 2. The name of node_type
  // 3. The max utility of this game
  // 4. Number of element of state
  // 5. The root player of mcts
  // 6. Total number of player of the game
  AotExtra *extra_aot = static_cast<AotExtra *>(extra);
  auto kernel_ptr = static_cast<CreationAttr *>(extra_aot->KernelData());
  std::string tree_name = kernel_ptr->tree_type;
  std::string node_name = kernel_ptr->node_type;
  float max_utility = kernel_ptr->max_utility;
  int state_size = static_cast<int>(kernel_ptr->state_size);
  int player = static_cast<int>(kernel_ptr->player);
  int total_num_player = static_cast<int>(kernel_ptr->total_num_player);
  // Input value
  // The input of MctsCreation will be treated as the global variable of the
  // monte carlo tree. It is shared by all the node in this monte carlo tree.
  // These variable will be saved in a std::vector with void* type. User can
  // call
  // MonteCarloTreeFactory::GetInstance().GetTreeVariableByHandle(tree_handle_)
  // to obtain the variable vector and select the corresponding variable by
  // index.
  void *device_state_ptr = nullptr;
  cudaMalloc(&device_state_ptr, sizeof(float) * (nparam - 1));
  float *input_global_const = static_cast<float *>(device_state_ptr);
  for (int i = 0; i < nparam - 1; i++) {
    cudaMemcpy(input_global_const + i, params[i], sizeof(float),
               cudaMemcpyDeviceToDevice);
  }
  // Output value
  // The output value of MctsCreation is the unique handle of this new monte
  // carlo tree.
  int64_t *output = static_cast<int64_t *>(params[nparam - 1]);

  int64_t tree_handle;
  MonteCarloTreePtr tree;
  std::tie(tree_handle, tree) = MonteCarloTreeFactory::GetInstance().CreateTree(
      tree_name, node_name, player, max_utility, state_size, total_num_player,
      input_global_const);
  if (tree == nullptr) {
    return kErrorCode;
  }
  tree->Memcpy(output, &tree_handle, sizeof(int64_t));
  cudaMemcpy(output, &tree_handle, sizeof(int64_t), cudaMemcpyHostToDevice);
  return 0;
}

class SelectionAttr : public AotKernelData {
public:
  float max_action;
  float tree_handle;
};

extern "C" int MctsSelectionInit(int *ndims, int64_t **shapes,
                                 const char **dtypes, AotExtra *extra) {
  SelectionAttr *kernel_ptr = new SelectionAttr;
  kernel_ptr->max_action = extra->Attr<float>("max_action");
  kernel_ptr->tree_handle = extra->Attr<float>("tree_handle");
  extra->SetKernelData(kernel_ptr);
  return 0;
}

extern "C" int MctsSelection(int nparam, void **params, int *ndims,
                             int64_t **shapes, const char **dtypes,
                             void *stream, void *extra) {
  // Input Attr
  // MctsSelection has 2 compulsory input attr
  // 1. The max length of action that MctsSelection returns, if max_action = -1,
  // it will only return the last action
  // 2. The unique tree handle
  AotExtra *extra_aot = static_cast<AotExtra *>(extra);
  auto kernel_ptr = static_cast<SelectionAttr *>(extra_aot->KernelData());
  int max_action = static_cast<int>(kernel_ptr->max_action);
  int64_t tree_handle = static_cast<int64_t>(kernel_ptr->tree_handle);
  // Output value
  // It has two output value:
  // 1. The handle of visited path, but its a dummy handle. There is no map to
  // represent the mapping between handle
  //    and visited path. The visited path is saved in tree. This dummy handle
  //    is more like an dummy object that user can operate in python side.
  // 2. If the max_action is given, it will return a Tensor which is combined by
  // actions of each node in visited path.
  //    Moreover, the Tensor will be filled with -1, if its length does not
  //    reach the max_action value. If the max_action is NOT given, it will only
  //    return the last action in visited path.
  int64_t *visited_path_handle = static_cast<int64_t *>(params[0]);
  int *out_action = static_cast<int *>(params[1]);

  auto tree = MonteCarloTreeFactory::GetInstance().GetTreeByHandle(tree_handle);
  if (tree == nullptr) {
    return kErrorCode;
  }
  int size_of_action = max_action;
  if (max_action == -1) {
    size_of_action = 1;
  }
  int *action_list =
      reinterpret_cast<int *>(tree->AllocateMem(size_of_action * sizeof(int)));
  tree->Memset(action_list, -1, sizeof(int) * size_of_action);
  auto ret = tree->Selection(action_list, max_action, stream);
  if (!ret) {
    return kErrorCode;
  }
  int64_t visited_handle = tree->placeholder_handle();
  tree->Memcpy(visited_path_handle, &visited_handle, sizeof(int64_t));
  tree->Memcpy(out_action, action_list, sizeof(int) * size_of_action);
  tree->Free(action_list);
  return 0;
}

class ExpansionAttr : public AotKernelData {
public:
  std::string node_type;
  bool has_init_reward;
  float tree_handle;
};

extern "C" int MctsExpansionInit(int *ndims, int64_t **shapes,
                                 const char **dtypes, AotExtra *extra) {
  ExpansionAttr *kernel_ptr = new ExpansionAttr;
  kernel_ptr->node_type = extra->Attr<std::string>("node_type");
  kernel_ptr->has_init_reward = extra->Attr<bool>("has_init_reward");
  kernel_ptr->tree_handle = extra->Attr<float>("tree_handle");
  extra->SetKernelData(kernel_ptr);
  return 0;
}

extern "C" int MctsExpansion(int nparam, void **params, int *ndims,
                             int64_t **shapes, const char **dtypes,
                             void *stream, void *extra) {
  // Input Attr
  // MctsExpansion has 3 compulsory input attr
  // 1. The name of node
  // 2. A indicator indicate whether node has init reward
  // 3. The unique tree handle
  AotExtra *extra_aot = static_cast<AotExtra *>(extra);
  auto kernel_ptr = static_cast<ExpansionAttr *>(extra_aot->KernelData());
  std::string node_name = kernel_ptr->node_type;
  bool has_init_reward = kernel_ptr->has_init_reward;
  int64_t tree_handle = static_cast<int64_t>(kernel_ptr->tree_handle);
  // Input value
  // MctsExpansion has 4 input values:
  // 1. A dummy handle, it is not used.
  // 2. action is a Tensor that is used to create the node
  // 3. prior is a Tensor that states for probability, which has the same length
  // as action.
  // 4. Which player does these nodes belong to
  int *action = static_cast<int *>(params[1]);
  float *prior = static_cast<float *>(params[2]);
  float *init_reward = static_cast<float *>(params[3]);
  // Output value
  // Whether expansion executes successfully.
  bool *output = static_cast<bool *>(params[4]);
  auto tree = MonteCarloTreeFactory::GetInstance().GetTreeByHandle(tree_handle);
  if (tree == nullptr) {
    return kErrorCode;
  }
  if (!has_init_reward) {
    init_reward = nullptr;
  }
  bool ret = tree->Expansion(node_name, action, prior, init_reward,
                             shapes[kInputIndex][0], tree->state_size());
  tree->Memcpy(output, &ret, sizeof(bool));
  return 0;
}

class BackpropagationAttr : public AotKernelData {
public:
  float tree_handle;
};

extern "C" int MctsBackpropagationInit(int *ndims, int64_t **shapes,
                                       const char **dtypes, AotExtra *extra) {
  BackpropagationAttr *kernel_ptr = new BackpropagationAttr;
  kernel_ptr->tree_handle = extra->Attr<float>("tree_handle");
  extra->SetKernelData(kernel_ptr);
  return 0;
}

extern "C" int MctsBackpropagation(int nparam, void **params, int *ndims,
                                   int64_t **shapes, const char **dtypes,
                                   void *stream, void *extra) {
  // Input Attr
  // MctsBackpropagation has 1 input attr
  // 1. The unique tree handle
  AotExtra *extra_aot = static_cast<AotExtra *>(extra);
  auto kernel_ptr = static_cast<BackpropagationAttr *>(extra_aot->KernelData());
  int64_t tree_handle = static_cast<int64_t>(kernel_ptr->tree_handle);
  // Input value
  // MctsBackpropagation has 3 input values:
  // 1. tree_handle is the unique tree handle.
  // 2. A dummy handle, it is not used.
  // 3. Returns that obtains from simulation is used to update all the nodes in
  // visited path.
  float *returns = static_cast<float *>(params[1]);
  // Output value
  // After backpropagation, whether the tree is fully solved.
  bool *output = static_cast<bool *>(params[2]);
  auto tree = MonteCarloTreeFactory::GetInstance().GetTreeByHandle(tree_handle);
  if (tree == nullptr) {
    return kErrorCode;
  }
  bool ret = tree->Backpropagation(returns, stream);
  // tree->Memcpy(output, &ret, sizeof(bool));
  cudaMemcpy(output, &ret, sizeof(bool), cudaMemcpyHostToDevice);
  return 0;
}

class BestActionAttr : public AotKernelData {
public:
  float tree_handle;
};

extern "C" int BestActionInit(int *ndims, int64_t **shapes, const char **dtypes,
                              AotExtra *extra) {
  BestActionAttr *kernel_ptr = new BestActionAttr;
  kernel_ptr->tree_handle = extra->Attr<float>("tree_handle");
  extra->SetKernelData(kernel_ptr);
  return 0;
}

extern "C" int BestAction(int nparam, void **params, int *ndims,
                          int64_t **shapes, const char **dtypes, void *stream,
                          void *extra) {
  // Input Attr
  // BestAction has 1 input attr
  // 1. The unique tree handle
  AotExtra *extra_aot = static_cast<AotExtra *>(extra);
  auto kernel_ptr = static_cast<BestActionAttr *>(extra_aot->KernelData());
  int64_t tree_handle = static_cast<int64_t>(kernel_ptr->tree_handle);
  // Output value
  // Return the best action.
  int *output = static_cast<int *>(params[0]);

  auto tree = MonteCarloTreeFactory::GetInstance().GetTreeByHandle(tree_handle);
  if (tree == nullptr) {
    return kErrorCode;
  }
  tree->Memcpy(output, tree->BestAction(), sizeof(int));
  return 0;
}

class UpdateLeafNodeOutcomeAttr : public AotKernelData {
public:
  float tree_handle;
};

extern "C" int UpdateLeafNodeOutcomeInit(int *ndims, int64_t **shapes,
                                         const char **dtypes, AotExtra *extra) {
  UpdateLeafNodeOutcomeAttr *kernel_ptr = new UpdateLeafNodeOutcomeAttr;
  kernel_ptr->tree_handle = extra->Attr<float>("tree_handle");
  extra->SetKernelData(kernel_ptr);
  return 0;
}

extern "C" int UpdateLeafNodeOutcome(int nparam, void **params, int *ndims,
                                     int64_t **shapes, const char **dtypes,
                                     void *stream, void *extra) {
  // Input Attr
  // UpdateLeafNodeOutcome has 1 input attr
  // 1. The unique tree handle
  AotExtra *extra_aot = static_cast<AotExtra *>(extra);
  auto kernel_ptr =
      static_cast<UpdateLeafNodeOutcomeAttr *>(extra_aot->KernelData());
  int64_t tree_handle = static_cast<int64_t>(kernel_ptr->tree_handle);
  // Input value
  // UpdateOutcome has 2 input values:
  // 1. A dummy handle, it is not used.
  // 2. The outcome of terminal state.
  float *outcome = static_cast<float *>(params[1]);
  int num_element = shapes[kInputIndex][0];
  float *outcome_host = new float[sizeof(float) * num_element];
  cudaMemcpy(outcome_host, outcome, sizeof(float) * num_element,
             cudaMemcpyDeviceToHost);
  // Output value
  // Whether update executes successfully.
  bool *output = static_cast<bool *>(params[2]);

  std::vector<float> return_value;
  for (int i = 0; i < num_element; i++) {
    return_value.emplace_back(outcome_host[i]);
  }
  auto tree = MonteCarloTreeFactory::GetInstance().GetTreeByHandle(tree_handle);
  if (tree == nullptr) {
    return kErrorCode;
  }
  int index = static_cast<int>(tree->visited_path().size() - 1);
  bool ret = tree->UpdateOutcome(return_value, index);
  tree->Memcpy(output, &ret, sizeof(bool));
  return 0;
}

class UpdateLeafNodeTerminalAttr : public AotKernelData {
public:
  float tree_handle;
};

extern "C" int UpdateLeafNodeTerminalInit(int *ndims, int64_t **shapes,
                                          const char **dtypes,
                                          AotExtra *extra) {
  UpdateLeafNodeTerminalAttr *kernel_ptr = new UpdateLeafNodeTerminalAttr;
  kernel_ptr->tree_handle = extra->Attr<float>("tree_handle");
  extra->SetKernelData(kernel_ptr);
  return 0;
}

extern "C" int UpdateLeafNodeTerminal(int nparam, void **params, int *ndims,
                                      int64_t **shapes, const char **dtypes,
                                      void *stream, void *extra) {
  // Input Attr
  // UpdateLeafNodeTerminal has 1 input attr
  // 1. The unique tree handle
  AotExtra *extra_aot = static_cast<AotExtra *>(extra);
  auto kernel_ptr =
      static_cast<UpdateLeafNodeTerminalAttr *>(extra_aot->KernelData());
  int64_t tree_handle = static_cast<int64_t>(kernel_ptr->tree_handle);
  // Input value
  // UpdateTerminal has 2 input values:
  // 1. A dummy handle, it is not used.
  // 2. The terminal state.
  bool *terminal = static_cast<bool *>(params[1]);
  bool *terminal_host = new bool[sizeof(bool)];
  cudaMemcpy(terminal_host, terminal, sizeof(bool), cudaMemcpyDeviceToHost);
  // Output value
  // Whether update executes successfully.
  bool *output = static_cast<bool *>(params[2]);

  auto tree = MonteCarloTreeFactory::GetInstance().GetTreeByHandle(tree_handle);
  if (tree == nullptr) {
    return kErrorCode;
  }
  int index = static_cast<int>(tree->visited_path().size() - 1);
  bool ret = tree->UpdateTerminal(*terminal_host, index);
  tree->Memcpy(output, &ret, sizeof(bool));
  return 0;
}

class UpdateLeafNodeStateAttr : public AotKernelData {
public:
  float tree_handle;
};

extern "C" int UpdateLeafNodeStateInit(int *ndims, int64_t **shapes,
                                       const char **dtypes, AotExtra *extra) {
  UpdateLeafNodeStateAttr *kernel_ptr = new UpdateLeafNodeStateAttr;
  kernel_ptr->tree_handle = extra->Attr<float>("tree_handle");
  extra->SetKernelData(kernel_ptr);
  return 0;
}

extern "C" int UpdateLeafNodeState(int nparam, void **params, int *ndims,
                                   int64_t **shapes, const char **dtypes,
                                   void *stream, void *extra) {
  // Input Attr
  // UpdateLeafNodeState has 1 input attr
  // 1. The unique tree handle
  AotExtra *extra_aot = static_cast<AotExtra *>(extra);
  auto kernel_ptr =
      static_cast<UpdateLeafNodeStateAttr *>(extra_aot->KernelData());
  int64_t tree_handle = static_cast<int64_t>(kernel_ptr->tree_handle);
  // Input value
  // UpdateState has 2 input values:
  // 1. A dummy handle, it is not used.
  // 2. State of environment
  float *state = static_cast<float *>(params[1]);
  // Output value
  // Whether update executes successfully.
  bool *output = static_cast<bool *>(params[2]);

  auto tree = MonteCarloTreeFactory::GetInstance().GetTreeByHandle(tree_handle);
  if (tree == nullptr) {
    return kErrorCode;
  }
  int index = static_cast<int>(tree->visited_path().size() - 1);
  bool ret = tree->UpdateState(state, index);
  tree->Memcpy(output, &ret, sizeof(bool));
  return 0;
}

class UpdateRootStateAttr : public AotKernelData {
public:
  float tree_handle;
};

extern "C" int UpdateRootStateInit(int *ndims, int64_t **shapes,
                                   const char **dtypes, AotExtra *extra) {
  UpdateRootStateAttr *kernel_ptr = new UpdateRootStateAttr;
  kernel_ptr->tree_handle = extra->Attr<float>("tree_handle");
  extra->SetKernelData(kernel_ptr);
  return 0;
}

extern "C" int UpdateRootState(int nparam, void **params, int *ndims,
                               int64_t **shapes, const char **dtypes,
                               void *stream, void *extra) {
  // Input Attr
  // UpdateRootState has 1 input attr
  // 1. The unique tree handle
  AotExtra *extra_aot = static_cast<AotExtra *>(extra);
  auto kernel_ptr = static_cast<UpdateRootStateAttr *>(extra_aot->KernelData());
  int64_t tree_handle = static_cast<int64_t>(kernel_ptr->tree_handle);
  // Input value
  // UpdateState has 1 input values:
  // 1. State of environment
  float *state = static_cast<float *>(params[0]);
  // Output value
  // Whether update executes successfully.
  bool *output = static_cast<bool *>(params[1]);

  auto tree = MonteCarloTreeFactory::GetInstance().GetTreeByHandle(tree_handle);
  if (tree == nullptr) {
    return kErrorCode;
  }
  tree->root()->set_state(state, tree->state_size());
  bool ret = true;
  tree->Memcpy(output, &ret, sizeof(bool));
  return 0;
}

class GetLastStateAttr : public AotKernelData {
public:
  float tree_handle;
};

extern "C" int GetLastStateInit(int *ndims, int64_t **shapes,
                                const char **dtypes, AotExtra *extra) {
  GetLastStateAttr *kernel_ptr = new GetLastStateAttr;
  kernel_ptr->tree_handle = extra->Attr<float>("tree_handle");
  extra->SetKernelData(kernel_ptr);
  return 0;
}

extern "C" int GetLastState(int nparam, void **params, int *ndims,
                            int64_t **shapes, const char **dtypes, void *stream,
                            void *extra) {
  // Input Attr
  // GetLastState has 1 input attr
  // 1. The unique tree handle
  AotExtra *extra_aot = static_cast<AotExtra *>(extra);
  auto kernel_ptr = static_cast<GetLastStateAttr *>(extra_aot->KernelData());
  int64_t tree_handle = static_cast<int64_t>(kernel_ptr->tree_handle);
  // Input value
  // GetState has 1 input values:
  // 1. A dummy handle, it is not used.
  // Output value
  // The state of the node that user specifies
  float *output = static_cast<float *>(params[1]);

  auto tree = MonteCarloTreeFactory::GetInstance().GetTreeByHandle(tree_handle);
  if (tree == nullptr) {
    return kErrorCode;
  }
  int index = static_cast<int>(tree->visited_path().size() - 2);
  if (index < 0) {
    index += static_cast<int>(tree->visited_path().size());
  }
  auto output_state = tree->GetState(index);
  tree->Memcpy(output, output_state, tree->state_size() * sizeof(float));
  return 0;
}

class DestroyTreeAttr : public AotKernelData {
public:
  float tree_handle;
};

extern "C" int DestroyTreeInit(int *ndims, int64_t **shapes,
                               const char **dtypes, AotExtra *extra) {
  DestroyTreeAttr *kernel_ptr = new DestroyTreeAttr;
  kernel_ptr->tree_handle = extra->Attr<float>("tree_handle");
  extra->SetKernelData(kernel_ptr);
  return 0;
}

extern "C" int DestroyTree(int nparam, void **params, int *ndims,
                           int64_t **shapes, const char **dtypes, void *stream,
                           void *extra) {
  // Input Value
  // Unique tree handle
  int64_t *tree_handle = static_cast<int64_t *>(params[0]);
  // Output Value
  // Whether restore success
  bool *output = static_cast<bool *>(params[1]);
  int64_t *tree_handle_host = new int64_t[sizeof(int64_t)];
  cudaMemcpy(tree_handle_host, tree_handle, sizeof(int64_t),
             cudaMemcpyDeviceToHost);
  auto tree =
      MonteCarloTreeFactory::GetInstance().GetTreeByHandle(*tree_handle_host);
  if (tree == nullptr) {
    return kErrorCode;
  }
  bool ret_tree =
      MonteCarloTreeFactory::GetInstance().DeleteTree(*tree_handle_host);
  // Delete Tree Variable
  if (!ret_tree) {
    return kErrorCode;
  }
  bool ret = true;
  tree->Memcpy(output, &ret, sizeof(bool));
  return 0;
}

class RestoreTreeAttr : public AotKernelData {
public:
  float tree_handle;
};

extern "C" int RestoreTreeInit(int *ndims, int64_t **shapes,
                               const char **dtypes, AotExtra *extra) {
  RestoreTreeAttr *kernel_ptr = new RestoreTreeAttr;
  kernel_ptr->tree_handle = extra->Attr<float>("tree_handle");
  extra->SetKernelData(kernel_ptr);
  return 0;
}

extern "C" int RestoreTree(int nparam, void **params, int *ndims,
                           int64_t **shapes, const char **dtypes, void *stream,
                           void *extra) {
  // Input Value
  // Unique tree handle
  int64_t *tree_handle = static_cast<int64_t *>(params[0]);
  // Output Value
  // Whether restore success
  bool *output = static_cast<bool *>(params[1]);
  int64_t *tree_handle_host = new int64_t[sizeof(int64_t)];
  cudaMemcpy(tree_handle_host, tree_handle, sizeof(int64_t),
             cudaMemcpyDeviceToHost);
  auto tree =
      MonteCarloTreeFactory::GetInstance().GetTreeByHandle(*tree_handle_host);
  if (tree == nullptr) {
    return kErrorCode;
  }
  tree->Restore();
  bool ret = true;
  tree->Memcpy(output, &ret, sizeof(bool));
}

class UpdateGlobalVariableAttr : public AotKernelData {
public:
  float tree_handle;
};

extern "C" int UpdateGlobalVariableInit(int *ndims, int64_t **shapes,
                                        const char **dtypes, AotExtra *extra) {
  UpdateGlobalVariableAttr *kernel_ptr = new UpdateGlobalVariableAttr;
  kernel_ptr->tree_handle = extra->Attr<float>("tree_handle");
  extra->SetKernelData(kernel_ptr);
  return 0;
}

extern "C" int UpdateGlobalVariable(int nparam, void **params, int *ndims,
                                    int64_t **shapes, const char **dtypes,
                                    void *stream, void *extra) {
  // Input Attr
  // UpdateRootState has 1 input attr
  // 1. The unique tree handle
  AotExtra *extra_aot = static_cast<AotExtra *>(extra);
  auto kernel_ptr =
      static_cast<UpdateGlobalVariableAttr *>(extra_aot->KernelData());
  int64_t tree_handle = static_cast<int64_t>(kernel_ptr->tree_handle);

  void *device_state_ptr = nullptr;
  cudaMalloc(&device_state_ptr, sizeof(float) * (nparam - 1));
  float *input_global_variable = static_cast<float *>(device_state_ptr);
  for (int i = 0; i < nparam - 1; i++) {
    cudaMemcpy(input_global_variable + i, params[i], sizeof(float),
               cudaMemcpyDeviceToDevice);
  }

  bool *output = static_cast<bool *>(params[nparam - 1]);
  MonteCarloTreeFactory::GetInstance().InsertGlobalVariable(
      tree_handle, input_global_variable);

  bool ret = true;
  cudaMemcpy(output, &ret, sizeof(bool), cudaMemcpyDeviceToDevice);
  return 0;
}

class GetRootInfoAttr : public AotKernelData {
public:
  float tree_handle;
};

extern "C" int GetRootInfoInit(int *ndims, int64_t **shapes,
                               const char **dtypes, AotExtra *extra) {
  GetRootInfoAttr *kernel_ptr = new GetRootInfoAttr;
  kernel_ptr->tree_handle = extra->Attr<float>("tree_handle");
  extra->SetKernelData(kernel_ptr);
  return 0;
}

extern "C" int GetRootInfo(int nparam, void **params, int *ndims,
                           int64_t **shapes, const char **dtypes, void *stream,
                           void *extra) {
  std::cout << "GPU does not support this function yet" << std::endl;
  return kErrorCode;
}
} // namespace utils
} // namespace mindspore_rl
