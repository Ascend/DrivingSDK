#include <graph/types.h>
#include <register/op_def.h>

#include <cstdint>

#include "bev_pool_tiling.h"
#include "ge/utils.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr size_t FEAT_IDX = 0;
constexpr size_t GEOM_FEAT_IDX = 1;
constexpr size_t INTERVAL_IDX_V1 = 3;
constexpr size_t INTERVAL_IDX_V2 = 6;
constexpr size_t B_IDX = 0;
constexpr size_t D_IDX = 1;
constexpr size_t H_IDX = 2;
constexpr size_t W_IDX = 3;
constexpr size_t C_IDX = 4;

constexpr int32_t TILING_ALIGN32B_FLAG = 1;
constexpr int32_t TILING_FP32_BIT = 1;
constexpr int32_t TILING_FP16_BIT = 2;
constexpr int32_t TILING_BF16_BIT = 3;

int32_t GetTilingKey(const ge::DataType dtype, optiling::BEVPoolTilingData& tiling)
{
    auto dtypeBytes = ge::GetSizeByDataType(dtype);
    int32_t cBytes = tiling.get_stride0() * dtypeBytes;
    int32_t key = cBytes % 32 == 0 ? TILING_ALIGN32B_FLAG : 0;
    switch (dtype) {
        case ge::DT_FLOAT:
            key |= 1 << TILING_FP32_BIT;
            break;
        case ge::DT_FLOAT16:
            key |= 1 << TILING_FP16_BIT;
            break;
        case ge::DT_BF16:
            key |= 1 << TILING_BF16_BIT;
            break;
        default:
            break; // here, fail-safe is not a good idea
    }
    return key;
}

enum Version {
    V1,
    V2
};
} // namespace

namespace optiling {
template<Version version>
static ge::graphStatus TilingForBEVPool(gert::TilingContext* context)
{
    CHECK_NULLPTR(context);
    BEVPoolTilingData tiling;
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto coreNum = platform.GetCoreNum();

    auto intervalShape =
        version == V1 ? context->GetInputShape(INTERVAL_IDX_V1) : context->GetInputShape(INTERVAL_IDX_V2);
    uint64_t nInterval = intervalShape->GetStorageShape().GetDim(0);

    uint64_t usedCoreNum = std::min(static_cast<uint64_t>(coreNum), nInterval);
    tiling.set_usedCoreNum(usedCoreNum);
    if (usedCoreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    auto avgTaskNum = nInterval / usedCoreNum;
    auto tailTaskNum = nInterval % usedCoreNum;
    tiling.set_totalTaskNum(nInterval);
    tiling.set_avgTaskNum(avgTaskNum);
    tiling.set_tailTaskNum(tailTaskNum);

    auto attrs = context->GetAttrs();
    if (!attrs) {
        return ge::GRAPH_FAILED;
    }
    auto getAttr = [attrs](size_t idx) -> uint64_t {
        auto ptr = attrs->GetInt(idx);
        if (!ptr) {
            return -1;
        }
        return static_cast<uint64_t>(*ptr);
    };
    auto b = getAttr(B_IDX);
    auto d = getAttr(D_IDX);
    auto h = getAttr(H_IDX);
    auto w = getAttr(W_IDX);
    auto c = getAttr(C_IDX);
    if (b < 0 || d < 0 || h < 0 || w < 0 || c < 0) {
        return ge::GRAPH_FAILED;
    }
    tiling.set_stride0(c);
    tiling.set_stride1(w * c);
    tiling.set_stride2(h * w * c);
    tiling.set_stride3(d * h * w * c);

    auto dtype = context->GetInputDesc(FEAT_IDX)->GetDataType();
    context->SetTilingKey(GetTilingKey(dtype, tiling));
    context->SetBlockDim(usedCoreNum);

    ADD_TILING_DATA(context, tiling)
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static graphStatus InferShapeForBEVPool(gert::InferShapeContext* context)
{
    auto attrs = context->GetAttrs();
    auto getAttr = [attrs](size_t idx) -> int64_t {
        auto ptr = attrs->GetInt(idx);
        if (!ptr) {
            return -1;
        }
        return static_cast<int64_t>(*ptr);
    };
    auto b = getAttr(B_IDX);
    auto d = getAttr(D_IDX);
    auto h = getAttr(H_IDX);
    auto w = getAttr(W_IDX);
    auto c = getAttr(C_IDX);
    if (b < 0 || d < 0 || h < 0 || w < 0 || c < 0) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* outShape = context->GetOutputShape(0);
    *outShape = {b, d, h, w, c};
    return GRAPH_SUCCESS;
}

static graphStatus InferShapeForBEVPoolGrad(gert::InferShapeContext* context)
{
    const gert::Shape* GeomFeatShape = context->GetInputShape(GEOM_FEAT_IDX);
    const auto n = GeomFeatShape->GetDim(0);
    auto attrs = context->GetAttrs();
    CHECK_NULLPTR(attrs)
    auto c = *attrs->GetInt(C_IDX);
    gert::Shape* gradFeatShape = context->GetOutputShape(0);
    *gradFeatShape = {n, c};
    return GRAPH_SUCCESS;
}

static graphStatus InferShapeForBEVPoolV2Grad(gert::InferShapeContext* context)
{
    gert::Shape* gradDepthShape = context->GetOutputShape(0);
    const gert::Shape* depthShape = context->GetInputShape(1);
    *gradDepthShape = *depthShape;
    gert::Shape* gradFeatShape = context->GetOutputShape(1);
    const gert::Shape* featShape = context->GetInputShape(2);
    *gradFeatShape = *featShape;
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class BEVPool : public OpDef {
public:
    explicit BEVPool(const char* name) : OpDef(name)
    {
        this->Input("feat")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("geom_feat")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("interval_lengths")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("interval_starts")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("b").AttrType(REQUIRED).Int();
        this->Attr("d").AttrType(REQUIRED).Int();
        this->Attr("h").AttrType(REQUIRED).Int();
        this->Attr("w").AttrType(REQUIRED).Int();
        this->Attr("c").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShapeForBEVPool);

        this->AICore().SetTiling(optiling::TilingForBEVPool<V1>);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

/**
 * @brief: BEVPoolGrad, the backward of bev_pool
 * @par Inputs:
 * grad_out: input grad, 5D tensor(b, d, h, w, c), dtype: float32, format:
 * NDHWC, ND geom_feat: input coords, 2D tensor(n, 4), dtype: int32, format: ND
 * interval_starts: starting position for pooled point, 1D tensor(n_interval),
 * dtype: int32, format: ND interval_lengths: the number of points in each
 * interval, 1D tensor(n_interval), dtype: int32, format: ND
 * @par Outputs:
 * grad_feat: output grad, 2D tensor(n, c), dtype: float32, format: ND
 * @par Attributes:
 * b: batch size, type: int
 * d: depth, type: int
 * w: width, type: int
 * h: height, type: int
 * n: number of points, type: int
 * c: channels, type: int
 **/
class BEVPoolGrad : public OpDef {
public:
    explicit BEVPoolGrad(const char* name) : OpDef(name)
    {
        this->Input("grad_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("geom_feat")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("interval_lengths")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("interval_starts")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("grad_feat")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("b").AttrType(REQUIRED).Int();
        this->Attr("d").AttrType(REQUIRED).Int();
        this->Attr("h").AttrType(REQUIRED).Int();
        this->Attr("w").AttrType(REQUIRED).Int();
        this->Attr("c").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShapeForBEVPoolGrad);

        this->AICore().SetTiling(optiling::TilingForBEVPool<V1>);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

class BEVPoolV2 : public OpDef {
public:
    explicit BEVPoolV2(const char* name) : OpDef(name)
    {
        this->Input("depth")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("feat")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("ranks_depth")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("ranks_feat")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("ranks_bev")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("interval_lengths")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("interval_starts")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("b").AttrType(REQUIRED).Int();
        this->Attr("d").AttrType(REQUIRED).Int();
        this->Attr("h").AttrType(REQUIRED).Int();
        this->Attr("w").AttrType(REQUIRED).Int();
        this->Attr("c").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShapeForBEVPool);

        this->AICore().SetTiling(optiling::TilingForBEVPool<V2>);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

/**
 * @brief: BEVPoolGrad, the backward of bev_pool
 * @par Inputs:
 * grad_out: input grad, 5D tensor(b, d, h, w, c), dtype: float32, format:
 * NDHWC, ND geom_feat: input coords, 2D tensor(n, 4), dtype: int32, format: ND
 * interval_starts: starting position for pooled point, 1D tensor(n_interval),
 * dtype: int32, format: ND interval_lengths: the number of points in each
 * interval, 1D tensor(n_interval), dtype: int32, format: ND
 * @par Outputs:
 * grad_feat: output grad, 2D tensor(n, c), dtype: float32, format: ND
 * @par Attributes:
 **/
class BEVPoolV2Grad : public OpDef {
public:
    explicit BEVPoolV2Grad(const char* name) : OpDef(name)
    {
        this->Input("grad_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("depth")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("feat")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("ranks_depth")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("ranks_feat")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("ranks_bev")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("interval_lengths")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("interval_starts")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("grad_depth")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("grad_feat")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("b").AttrType(REQUIRED).Int();
        this->Attr("d").AttrType(REQUIRED).Int();
        this->Attr("h").AttrType(REQUIRED).Int();
        this->Attr("w").AttrType(REQUIRED).Int();
        this->Attr("c").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShapeForBEVPoolV2Grad);

        this->AICore().SetTiling(optiling::TilingForBEVPool<V2>);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(BEVPool);
OP_ADD(BEVPoolGrad);
OP_ADD(BEVPoolV2);
OP_ADD(BEVPoolV2Grad);
} // namespace ops
