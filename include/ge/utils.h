// Copyright (c) 2024 Huawei Technologies Co., Ltd
// Copyright (c) 2019, Facebook CORPORATION.
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

#ifndef GE_UTILS_H_
#define GE_UTILS_H_

#define CHECK_NULLPTR(ptr)       \
    if (ptr == nullptr) {        \
        return ge::GRAPH_FAILED; \
    }

#define ADD_TILING_DATA(contex, tiling)                                                                      \
    CHECK_NULLPTR(context->GetRawTilingData())                                                               \
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity()); \
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

template<typename T1, typename T2>
inline T1 Ceil(const T1& x, const T2& y)
{
    if (y == 0) {
        return 0;
    }
    return (x + y - 1) / y;
}

template<typename T1, typename T2>
inline T1 AlignUp(const T1& x, const T2& y)
{
    if (y == 0) {
        return 0;
    }
    return ((x + y - 1) / y) * y;
}

template<typename T1, typename T2>
inline T1 Tail(const T1& x, const T2& y)
{
    if (x == 0 || y == 0) {
        return 0;
    }
    return (x - 1) % y + 1;
}
#endif // GE_UTILS_H_