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

#ifndef MINDSPORE_RL_UTILS_MCTS_GPU_CUDA_IMPL_VANILLA_IMPL_CUH_
#define MINDSPORE_RL_UTILS_MCTS_GPU_CUDA_IMPL_VANILLA_IMPL_CUH_

#include <cuda_runtime_api.h>

void CalSelectionPolicy(int *explore_count, float *total_reward, int *parent_explore_count, float *uct_ptr,
                        float *uct_value, cudaStream_t cuda_stream);

void CalUpdate(int *explore_count, float *total_reward, float *values, int player, cudaStream_t cuda_stream);

#endif  // MINDSPORE_RL_UTILS_MCTS_GPU_CUDA_IMPL_VANILLA_IMPL_CUH_
