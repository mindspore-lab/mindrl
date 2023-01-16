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

#include "argmax_impl.cuh"
#include <stdio.h>

__global__ void Argmax(const float *selection_value, int num_items, int *output) {
  int max_index = 0;
  float max_value = selection_value[0];
  for (int i = 1; i < num_items; i++) {
    if (selection_value[i] > max_value) {
      max_value = selection_value[i];
      max_index = i;
    }
  }
  output[0] = max_index;
  return;
}

void CalArgmax(const float *selection_value, int num_items, int *output, cudaStream_t cuda_stream) {
  dim3 blockSize(1);
  dim3 gridSize(1);
  Argmax<<<gridSize, blockSize, 0, cuda_stream>>>(selection_value, num_items, output);
  return;
}
