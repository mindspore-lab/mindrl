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

#ifndef MINDSPORE_RL_UTILS_MCTS_LOGGING_H_
#define MINDSPORE_RL_UTILS_MCTS_LOGGING_H_

#include <stdio.h>
#include <cstdarg>
namespace mindspore_rl {
namespace utils {
constexpr int MAX_LOG_LEN = 500;
constexpr int ONE = 1;

void LogModule(char *buf, int bufLen, const char *fmt, ...) {
  (void)bufLen;
  va_list ap;
  va_start(ap, fmt);

  int iRet = vsnprintf(buf, MAX_LOG_LEN, fmt, ap);
  if (iRet < 0) {
    printf("_Log vsnprintf_s failed.\n");
  }
  va_end(ap);
}

#define LOG_ERROR(fmt, ...)                                                    \
  do {                                                                         \
    char Logbuff[MAX_LOG_LEN] = {0};                                           \
    LogModule(Logbuff, sizeof(Logbuff), fmt, ##__VA_ARGS__);                   \
    printf("[ERROR] [mindspore_rl/%s:%d] %s] %s\n", __FILE__, __LINE__,        \
           __FUNCTION__, Logbuff);                                             \
  } while (0)
} // namespace utils
} // namespace mindspore_rl
#endif // MINDSPORE_RL_UTILS_MCTS_LOGGING_H_
