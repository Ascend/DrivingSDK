/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
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

/*!
 * \file furthest_point_sampling.cc
 * \brief
 */
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"
#include "furthest_point_sampling_tiling.h"

using namespace ge;
using namespace std;
using namespace AscendC;

namespace optiling {
/****************constexpr definition*****************/
constexpr int64_t FP32_MODE = 0;
constexpr int64_t POINTSDIMSNUM = 3;

/****************struct definition*****************/
struct ub_memory_tag {
    uint64_t ub_size = 0;
    uint64_t ub_reserve = 3072;
    // 8 :dev into 8 pieces(point_x, point_y, point_z, temp, distance, pointTempX, pointTempY, pointTempZ, store N data each)
    uint64_t ub_data_blocks = 8;
};

/****************function definition*****************/
template<typename T>
inline T getSmallestMulVal(T data, T multiple)
{
    if (multiple == 0) {
        return 0;
    }
    return ((data + multiple - 1) / multiple);
}

template<typename T>
inline T getSmallestMul(T data, T multiple)
{
    return (getSmallestMulVal(data, multiple) * multiple);
}

/****************class definition*****************/
class FurthestPointSamplingTiling {
public:
    explicit FurthestPointSamplingTiling(gert::TilingContext* context) : TilingContext(context) {};
    ge::graphStatus Init();
    ge::graphStatus RunKernelTiling();
private:
    inline void SetTilingKeyMode(ge::DataType dType);
    inline uint64_t UbBlocksDataSpace(uint64_t data_num);
    inline uint64_t UbBlocksWorkSpace(uint64_t data_num);
    inline uint64_t UbBlocksSpace(uint64_t data_num);
    inline uint64_t FindMaxDataBlock();
    // in Kernel, we use ReduceMax requiring us to calc size of worklocal
    inline uint32_t calcWorkLocalSize(uint32_t max_repeat_times);
private:
    FurthestPointSamplingTilingData TilingData;
    gert::TilingContext* TilingContext = nullptr;

    uint32_t coreNum;

    uint32_t batch;
    uint32_t N;
    uint32_t numPoints;
    uint32_t pieces;
    uint32_t formerNum;
    uint32_t tailNum;
    uint32_t workSize;
    uint32_t idxTempSize;
    uint32_t bigCoreBatch;
    uint32_t smallCoreBatch;
    uint32_t bigCoreNum;
    uint32_t repeats;

    ub_memory_tag ub_memory;

    uint64_t point_dtype_size;
};

/****************class impl*****************/
ge::graphStatus FurthestPointSamplingTiling::Init()
{
    const gert::StorageShape *point_xyz_shape = TilingContext->GetInputShape(0);
    const gert::RuntimeAttrs *attrs           = TilingContext->GetAttrs();
    uint64_t                  max_data_num;

    auto platformInfoPtr = TilingContext->GetPlatformInfo();
    if ((platformInfoPtr == nullptr) || (point_xyz_shape == nullptr) || (attrs == nullptr)) {
        return ge::GRAPH_FAILED;
    }
    if (attrs->GetAttrPointer<uint32_t>(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto platformInfo = platform_ascendc::PlatformAscendC(platformInfoPtr);

    if (TilingContext->GetInputDesc(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // Set Tiling Key
    SetTilingKeyMode(TilingContext->GetInputDesc(0)->GetDataType());

    // get core num
    this->coreNum = platformInfo.GetCoreNumAiv();
    if (this->coreNum == 0) {
        return ge::GRAPH_FAILED;
    }

    // get ub_size，cal the capability that is aligned with 256 bytes
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, this->ub_memory.ub_size);

    // Get input args
    if (point_xyz_shape->GetStorageShape().GetDimNum() != optiling::POINTSDIMSNUM) {
        return ge::GRAPH_FAILED;
    }
    this->batch     = point_xyz_shape->GetStorageShape().GetDim(0);
    this->N         = point_xyz_shape->GetStorageShape().GetDim(2);
    this->numPoints = *(attrs->GetAttrPointer<uint32_t>(0));

    // get the capability on UB
    max_data_num = FindMaxDataBlock(); // pieces, repeats, workSize calc in this func

    if (this->repeats == 0) {
        return ge::GRAPH_FAILED;
    }

    // Tiling Args calc
    this->bigCoreBatch   = getSmallestMulVal<uint32_t>(this->batch, this->coreNum);
    this->smallCoreBatch = this->batch / this->coreNum;
    if (this->bigCoreBatch == this->smallCoreBatch) {
        this->bigCoreNum = this->coreNum;
    } else if ((this->bigCoreBatch == 1) && (this->smallCoreBatch == 0)) {
        this->bigCoreNum = this->batch;
    } else {
        this->bigCoreNum = (this->batch - (this->smallCoreBatch * this->coreNum)) /
            (this->bigCoreBatch - this->smallCoreBatch);
    }

    this->formerNum = ((this->repeats * 256) / this->point_dtype_size);
    this->tailNum   = this->N - this->formerNum * (this->pieces - 1);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FurthestPointSamplingTiling::RunKernelTiling()
{
    size_t sysWorkspaceSize = 16 * 1024 * 1024; // Alloc 16M workspace
    size_t userWorkSpaceSize = this->batch * this->N * this->point_dtype_size; // NearestDist needs a space to be moved out
    size_t *currentWorkSpace = TilingContext->GetWorkspaceSizes(1);
    if (currentWorkSpace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkSpace[0] = userWorkSpaceSize + sysWorkspaceSize;

    TilingData.set_batch(this->batch);
    TilingData.set_N(this->N);
    TilingData.set_numPoints(this->numPoints);
    TilingData.set_pieces(this->pieces);
    TilingData.set_formerNum(this->formerNum);
    TilingData.set_tailNum(this->tailNum);
    TilingData.set_workSize(this->workSize);
    TilingData.set_idxTempSize(this->idxTempSize);
    TilingData.set_bigCoreBatch(this->bigCoreBatch);
    TilingData.set_smallCoreBatch(this->smallCoreBatch);
    TilingData.set_bigCoreNum(this->bigCoreNum);
    if (this->batch <= this->coreNum) {
        TilingContext->SetBlockDim(this->batch);
    } else {
        TilingContext->SetBlockDim(this->coreNum);
    }
    TilingData.set_repeats(this->repeats);

    if (TilingContext->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    TilingData.SaveToBuffer(TilingContext->GetRawTilingData()->GetData(), TilingContext->GetRawTilingData()->GetCapacity());
    TilingContext->GetRawTilingData()->SetDataSize(TilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

inline uint32_t FurthestPointSamplingTiling::calcWorkLocalSize(uint32_t max_repeat_times)
{
    uint32_t elemPerBlock     = 32 / this->point_dtype_size; // num of data one block can store
    uint32_t elemPerRepeat    = 256 / this->point_dtype_size; // num of data one repeat can deal
    
    uint32_t iter1OutputCount = max_repeat_times * 2; // num of temp data in 1st stage
    uint32_t iter2AlignStart  = getSmallestMul<uint32_t>(iter1OutputCount, elemPerBlock); // align with 32 bytes
    uint32_t iter2OutputCount = getSmallestMulVal<uint32_t>(iter1OutputCount, elemPerRepeat) * 2; // num of temp data in 2nd stage
    uint32_t iter3AlignStart  = getSmallestMul<uint32_t>(iter2OutputCount, elemPerBlock); // align with 32 bytes
    uint32_t iter3OutputCount = getSmallestMulVal<uint32_t>(iter1OutputCount, elemPerRepeat) * 2; // num of temp data in 3rd stage
    uint32_t iter3AlignEnd    = getSmallestMul<uint32_t>(iter2OutputCount, elemPerBlock); // align with 32 bytes

    uint32_t finalWorkLocalNeedSize = iter2AlignStart + iter3AlignStart + iter3AlignEnd;
    uint32_t totalBytes             = finalWorkLocalNeedSize * this->point_dtype_size;

    if (totalBytes % 32 != 0) {
        return ge::GRAPH_FAILED;
    }
    return totalBytes;
}

inline uint64_t FurthestPointSamplingTiling::FindMaxDataBlock()
{
    // divide & conquer ==> find the capability, if bigger than N, split then cal
    uint64_t M = this->ub_memory.ub_size - this->ub_memory.ub_reserve;
    uint64_t low = 1; // at least there exits one data in UB
    uint64_t high = M / this->point_dtype_size;
    uint64_t max_data_num = 0;

    while (low <= high) {
        uint64_t mid = low + (high - low) / 2;
        if (UbBlocksSpace(mid) <= M) {
            max_data_num = mid;
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    return max_data_num;
}

inline void FurthestPointSamplingTiling::SetTilingKeyMode(ge::DataType dType)
{
    switch (dType) {
        case ge::DT_FLOAT:
            TilingContext->SetTilingKey(FP32_MODE);
            this->point_dtype_size = 4; // 4: float32, 4 bytes
            break;
        default:
            TilingContext->SetTilingKey(FP32_MODE);
            this->point_dtype_size = 4; // 4: float32, 4 bytes
            break;
    }
}

inline uint64_t FurthestPointSamplingTiling::UbBlocksDataSpace(uint64_t data_num)
{
    // data type is the same among the first 5 blocks, the num is data_num, aligned with 256 bytes
    return getSmallestMul<uint64_t>(this->point_dtype_size * data_num, 256);
}

inline uint64_t FurthestPointSamplingTiling::UbBlocksWorkSpace(uint64_t data_num)
{
    uint64_t singleBlockRepeats = getSmallestMulVal<uint64_t>(this->point_dtype_size * data_num, 256);
    uint64_t totalRepeats = getSmallestMulVal<uint64_t>(this->point_dtype_size * this->N, 256);

    this->pieces = (uint32_t)getSmallestMulVal<uint64_t>(totalRepeats, singleBlockRepeats);
    this->repeats = (singleBlockRepeats < totalRepeats) ? ((uint32_t)singleBlockRepeats) : ((uint32_t)totalRepeats);
    return (uint64_t)calcWorkLocalSize(this->repeats);
}

inline uint64_t FurthestPointSamplingTiling::UbBlocksSpace(uint64_t data_num)
{
    uint64_t dataSpace   = UbBlocksDataSpace(data_num);
    uint64_t workSpace   = UbBlocksWorkSpace(data_num);

    this->workSize = workSpace;
    this->idxTempSize = getSmallestMul<uint32_t>(this->pieces * 2 * this->point_dtype_size, 32);

    return (this->ub_memory.ub_data_blocks * dataSpace + workSpace + this->idxTempSize);
}

/****************main body*****************/
static ge::graphStatus TilingFurthestPointSampling(gert::TilingContext* context)
{
    FurthestPointSamplingTiling tilingObject(context);
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    tilingObject.Init();
    return tilingObject.RunKernelTiling();
}
}

namespace ge {
static ge::graphStatus InfershapeForFurthestPointSampling(gert::InferShapeContext *context)
{
    const gert::Shape        *point_xyz_shape = context->GetInputShape(0);
    const gert::RuntimeAttrs *attrs           = context->GetAttrs();
    gert::Shape              *index_shape     = context->GetOutputShape(0);
    if ((point_xyz_shape == nullptr) || (attrs == nullptr) || (index_shape == nullptr)) {
        return ge::GRAPH_FAILED;
    }
    if (attrs->GetAttrPointer<int32_t>(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }

    if (point_xyz_shape->GetDimNum() != optiling::POINTSDIMSNUM) {
        return ge::GRAPH_FAILED;
    }
    uint32_t batch      = point_xyz_shape->GetDim(0);
    uint32_t N          = point_xyz_shape->GetDim(2);
    uint32_t num_points = *(attrs->GetAttrPointer<int32_t>(0));

    index_shape->SetDimNum(2);
    index_shape->SetDim(0, batch);
    index_shape->SetDim(1, num_points);

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForFurthestPointSampling(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_INT32);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class FurthestPointSampling : public OpDef {
public:
    explicit FurthestPointSampling(const char* name) : OpDef(name)
    {
        this->Input("point_xyz")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("nearest_temp")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("num_points")
            .AttrType(REQUIRED)
            .Int();

        this->SetInferShape(ge::InfershapeForFurthestPointSampling)
            .SetInferDataType(ge::InferDataTypeForFurthestPointSampling);
        this->AICore().SetTiling(optiling::TilingFurthestPointSampling);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true);
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
    }
};

OP_ADD(FurthestPointSampling);
}