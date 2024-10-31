// Copyright (c) 2024 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#ifndef LOG_LOG_H_
#define LOG_LOG_H_

#include <acl/acl_base.h>
#include <stdlib.h>

namespace mx_driving {
namespace log {

inline bool IsACLGlobalLogOn(aclLogLevel level)
{
    const static int getACLGlobalLogLevel = []() -> int {
        char* env_val = std::getenv("ASCEND_GLOBAL_LOG_LEVEL");
        int64_t envFlag = (env_val != nullptr) ? strtol(env_val, nullptr, 10) : ACL_ERROR;
        return static_cast<int>(envFlag);
    }();
    return (getACLGlobalLogLevel <= level);
}
} // namespace log
} // namespace mx_driving

#define MX_DRIVING_LOGE(fmt, ...)                                                       \
    do {                                                                                \
        if (mx_driving::log::IsACLGlobalLogOn(ACL_ERROR)) {                             \
            aclAppLog(ACL_ERROR, __FILE__, __FUNCTION__, __LINE__, fmt, ##__VA_ARGS__); \
        }                                                                               \
    } while (0);
#define MX_DRIVING_LOGW(fmt, ...)                                                         \
    do {                                                                                  \
        if (mx_driving::log::IsACLGlobalLogOn(ACL_WARNING)) {                             \
            aclAppLog(ACL_WARNING, __FILE__, __FUNCTION__, __LINE__, fmt, ##__VA_ARGS__); \
        }                                                                                 \
    } while (0);
#define MX_DRIVING_LOGI(fmt, ...)                                                      \
    do {                                                                               \
        if (mx_driving::log::IsACLGlobalLogOn(ACL_INFO)) {                             \
            aclAppLog(ACL_INFO, __FILE__, __FUNCTION__, __LINE__, fmt, ##__VA_ARGS__); \
        }                                                                              \
    } while (0);
#define MX_DRIVING_LOGD(fmt, ...)                                                       \
    do {                                                                                \
        if (mx_driving::log::IsACLGlobalLogOn(ACL_DEBUG)) {                             \
            aclAppLog(ACL_DEBUG, __FILE__, __FUNCTION__, __LINE__, fmt, ##__VA_ARGS__); \
        }                                                                               \
    } while (0);
#endif // LOG_LOG_H_
