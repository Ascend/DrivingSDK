// Copyright (c) 2023 Huawei Technologies Co., Ltd
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

#include "csrc/OpApiCommon.h"

#include <pwd.h>
#include <sys/stat.h>

thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset = 0;
constexpr int g_mix64Shift = 33;

typedef void (*AddTensorAddrToCachedList)(void* addr);

void AddParamToBuf(const at::Tensor& at_tensor)
{
    static const auto addTensorAddrToCachedListAddr = GetOpApiFuncAddr("AddTensorAddrToCachedList");
    AddTensorAddrToCachedList addTensorAddrToCachedListFunc =
        reinterpret_cast<AddTensorAddrToCachedList>(addTensorAddrToCachedListAddr);
    if (!at_tensor.defined()) {
        MEMCPY_TO_BUF(",", 1);
        return;
    }
    // view shape
    MEMCPY_TO_BUF(at_tensor.sizes().data(), at_tensor.sizes().size() * sizeof(int64_t));
    // data type
    auto st = at_tensor.scalar_type();
    MEMCPY_TO_BUF(&st, sizeof(st));
    // seperator
    MEMCPY_TO_BUF(",", 1);
    // strides
    MEMCPY_TO_BUF(at_tensor.strides().data(), at_tensor.sizes().size() * sizeof(int64_t));
    // offset
    auto so = at_tensor.storage_offset();
    MEMCPY_TO_BUF(&so, sizeof(so));
    // storage shape
    aclDataType acl_data_type = kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(st)];
    c10::SmallVector<int64_t, 5> storageDims;
    if (acl_data_type != ACL_STRING) {
        storageDims.push_back(at_tensor.storage().nbytes() / at_tensor.itemsize());
    }
    MEMCPY_TO_BUF(storageDims.data(), storageDims.size() * sizeof(int64_t));

    addTensorAddrToCachedListFunc(const_cast<void*>(at_tensor.storage().data()));
}

void AddParamToBuf(const at::Scalar& at_scalar)
{
    at::ScalarType scalar_data_type = at_scalar.type();
    switch (scalar_data_type) {
        case at::ScalarType::Double: {
            double value = at_scalar.toDouble();
            MEMCPY_TO_BUF(&value, sizeof(double));
            break;
        }
        case at::ScalarType::Long: {
            int64_t value = at_scalar.toLong();
            MEMCPY_TO_BUF(&value, sizeof(int64_t));
            break;
        }
        case at::ScalarType::Bool: {
            bool value = at_scalar.toBool();
            MEMCPY_TO_BUF(&value, sizeof(bool));
            break;
        }
        case at::ScalarType::ComplexDouble: {
            auto value = at_scalar.toComplexDouble();
            MEMCPY_TO_BUF(&value, sizeof(value));
            break;
        }
        default: {
            break;
        }
    }
}

void AddParamToBuf(const at::IntArrayRef& at_array)
{
    MEMCPY_TO_BUF(at_array.data(), at_array.size() * sizeof(int64_t));
}

void AddParamToBuf(const at::ArrayRef<bool>& at_array)
{
    MEMCPY_TO_BUF(at_array.data(), at_array.size() * sizeof(bool));
}

void AddParamToBuf(const at::TensorList& at_tensor_list)
{
    for (size_t i = 0; i < at_tensor_list.size(); i++) {
        AddParamToBuf(at_tensor_list[i]);
    }
    auto counter = at_tensor_list.size();
    MEMCPY_TO_BUF(&counter, sizeof(counter));
}

void AddParamToBuf(const c10::optional<at::Tensor>& opt_tensor)
{
    if (opt_tensor.has_value() && opt_tensor.value().defined()) {
        AddParamToBuf(opt_tensor.value());
    } else {
        MEMCPY_TO_BUF(",", 1);
    }
}

void AddParamToBuf(const c10::optional<at::IntArrayRef>& opt_array)
{
    if (opt_array.has_value()) {
        AddParamToBuf(opt_array.value());
    } else {
        MEMCPY_TO_BUF(",", 1);
    }
}

void AddParamToBuf(const c10::optional<at::Scalar>& opt_scalar)
{
    if (opt_scalar.has_value()) {
        AddParamToBuf(opt_scalar.value());
    } else {
        MEMCPY_TO_BUF(",", 1);
    }
}

void AddParamToBuf(const at::ScalarType scalar_type)
{
    MEMCPY_TO_BUF(&scalar_type, sizeof(scalar_type));
}

void AddParamToBuf(const string& s)
{
    MEMCPY_TO_BUF(s.c_str(), s.size());
}

void AddParamToBuf() {}

inline uint64_t rotl64(uint64_t x, int8_t r)
{
    return (x << r) | (x >> (64 - r));
}

#define ROTL64(x, y)    rotl64(x, y)
#define BIG_CONSTANT(x) (x##LLU)

inline uint64_t GetBlock64(const uint64_t* p, int i)
{
    return p[i];
}

inline uint64_t fmix64(uint64_t k)
{
    // 0xff51afd7ed558ccd and 0xc4ceb9fe1a85ec53 are carefully selected constants to allow
    // hash values to be more evenly distributed in 64-bit space after multiplication.
    k ^= k >> g_mix64Shift;
    k *= BIG_CONSTANT(0xff51afd7ed558ccd);
    k ^= k >> g_mix64Shift;
    k *= BIG_CONSTANT(0xc4ceb9fe1a85ec53);
    k ^= k >> g_mix64Shift;

    return k;
}

uint64_t MurmurHash(const void* key, const int len, const uint32_t seed = 0xdeadb0d7)
{
    const uint8_t* data = (const uint8_t*)key;
    // the length of each block is 16 bytes
    const int nblocks = len / 16;
    uint64_t h1 = seed;
    uint64_t h2 = seed;

    // 0x87c37b91114253d5 and 0x4cf5ad432745937f are carefully selected constants to
    // blocking and obfuscation of input data
    const uint64_t c1 = BIG_CONSTANT(0x87c37b91114253d5);
    const uint64_t c2 = BIG_CONSTANT(0x4cf5ad432745937f);

    const uint64_t* blocks = (const uint64_t*)(data);

    for (int i = 0; i < nblocks; i++) {
        int even_num = 2;
        int odd_num = 1;
        uint64_t k1 = GetBlock64(blocks, i * even_num);
        uint64_t k2 = GetBlock64(blocks, i * even_num + odd_num);

        int8_t k1_shift = 31;
        k1 *= c1;
        k1 = ROTL64(k1, k1_shift);
        k1 *= c2;
        h1 ^= k1;

        int8_t h1_shift = 27;
        h1 = ROTL64(h1, h1_shift);
        h1 += h2;
        // increase randomness by mul by 5 and adding a constant
        h1 = h1 * 5 + 0x52dce729;

        int8_t k2_shift = 33;
        k2 *= c2;
        k2 = ROTL64(k2, k2_shift);
        k2 *= c1;
        h2 ^= k2;

        int8_t h2_shift = 31;
        h2 = ROTL64(h2, h2_shift);
        h2 += h1;
        // increase randomness by mul by 5 and adding a constant
        h2 = h2 * 5 + 0x38495ab5;
    }

    // the length of each block is 16 bytes
    const uint8_t* tail = (const uint8_t*)(data + nblocks * 16);
    uint64_t k1 = 0;
    uint64_t k2 = 0;
    // because the size of a block is 16, different offsets are calculated for tail blocks
    // for different sizes
    switch (len & 15) {
        case 15:
            k2 ^= ((uint64_t)tail[14]) << 48;
            [[fallthrough]];
        case 14:
            k2 ^= ((uint64_t)tail[13]) << 40;
            [[fallthrough]];
        case 13:
            k2 ^= ((uint64_t)tail[12]) << 32;
            [[fallthrough]];
        case 12:
            k2 ^= ((uint64_t)tail[11]) << 24;
            [[fallthrough]];
        case 11:
            k2 ^= ((uint64_t)tail[10]) << 16;
            [[fallthrough]];
        case 10:
            k2 ^= ((uint64_t)tail[9]) << 8;
            [[fallthrough]];
        case 9:
            k2 ^= ((uint64_t)tail[8]) << 0;
            k2 *= c2;
            k2 = ROTL64(k2, 33);
            k2 *= c1;
            h2 ^= k2;
            [[fallthrough]];
        case 8:
            k1 ^= ((uint64_t)tail[7]) << 56;
            [[fallthrough]];
        case 7:
            k1 ^= ((uint64_t)tail[6]) << 48;
            [[fallthrough]];
        case 6:
            k1 ^= ((uint64_t)tail[5]) << 40;
            [[fallthrough]];
        case 5:
            k1 ^= ((uint64_t)tail[4]) << 32;
            [[fallthrough]];
        case 4:
            k1 ^= ((uint64_t)tail[3]) << 24;
            [[fallthrough]];
        case 3:
            k1 ^= ((uint64_t)tail[2]) << 16;
            [[fallthrough]];
        case 2:
            k1 ^= ((uint64_t)tail[1]) << 8;
            [[fallthrough]];
        case 1:
            k1 ^= ((uint64_t)tail[0]) << 0;
            k1 *= c1;
            k1 = ROTL64(k1, 31);
            k1 *= c2;
            h1 ^= k1;
            [[fallthrough]];
        default:
            break;
    };

    h1 ^= len;
    h2 ^= len;

    h1 += h2;
    h2 += h1;

    h1 = fmix64(h1);
    h2 = fmix64(h2);

    h1 += h2;
    h2 += h1;
    return h2;
}

uint64_t CalcHashId()
{
    if (g_hashOffset == kHashBufMaxSize) {
        return 0;
    }
    uint64_t hash_id = MurmurHash(g_hashBuf, g_hashOffset);
    return hash_id;
}
