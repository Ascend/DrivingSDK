#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ge/utils.h"
#include "tiling/tiling_api.h"
#include "cylinder_query_tiling.h"

#define Ceil32(num) (((num) + 31) / 32 * 32)

namespace {
    constexpr uint32_t FLOAT_BYTE_SIZE = 4;

    constexpr uint32_t GROUP_IDX_SHAPE_DIM = 3;

    // 输入Tensor的下标
    constexpr uint32_t INPUT_NEW_XYZ_IDX = 0;

    // Attr下标
    const size_t BATCH_SIZE_INDEX = 0;
    const size_t POINT_CLOUD_SIZE_INDEX = 1;
    const size_t QUERY_POINT_SIZE_INDEX = 2;
    const size_t RADIUS_INDEX = 3;
    const size_t HMIN_INDEX = 4;
    const size_t HMAX_INDEX = 5;
    const size_t NSAMPLE_INDEX = 6;

    // 输出Tensor的下标
    constexpr uint32_t OUTPUT_QUERY_RES_IDX = 0;

    // 最小数据块中的点数 / 字节数
    constexpr uint32_t BLOCK_POINT_SIZE = 8;
    constexpr uint32_t BLOCK_BYTE_SIZE = 96;
}

namespace ge {
static ge::graphStatus InferShapeForCylinderQuery(gert::InferShapeContext* context)
{
    const gert::RuntimeAttrs *attr = context->GetAttrs();
    if (attr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape *groupIdxShape = context->GetOutputShape(OUTPUT_QUERY_RES_IDX);
    if (groupIdxShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto batchPtr = attr->GetAttrPointer<uint32_t>(BATCH_SIZE_INDEX);
    auto queryPointSize = attr->GetAttrPointer<uint32_t>(QUERY_POINT_SIZE_INDEX);
    auto nsamplePtr = attr->GetAttrPointer<uint32_t>(NSAMPLE_INDEX);
    auto pointCloudSizePtr = attr->GetAttrPointer<uint32_t>(POINT_CLOUD_SIZE_INDEX);

    if (!batchPtr || !queryPointSize || !nsamplePtr || !pointCloudSizePtr) {
        return ge::GRAPH_FAILED;
    }

    groupIdxShape->SetDimNum(GROUP_IDX_SHAPE_DIM);
    *groupIdxShape = {*batchPtr, *queryPointSize, *pointCloudSizePtr};

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForCylinderQuery(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(OUTPUT_QUERY_RES_IDX, ge::DataType::DT_FLOAT);
    return GRAPH_SUCCESS;
}
}

namespace optiling {
static ge::graphStatus TilingForCylinderQuery(gert::TilingContext* context)
{
    CylinderQueryTilingData tiling;

    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 硬件信息
    auto platformInfo = context->GetPlatformInfo();
    CHECK_NULLPTR(platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    // 输入属性
    const gert::RuntimeAttrs *attr = context->GetAttrs(); // 属性
    if (attr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    
    uint32_t B = *(attr->GetAttrPointer<uint32_t>(BATCH_SIZE_INDEX));
    uint32_t N = *(attr->GetAttrPointer<uint32_t>(POINT_CLOUD_SIZE_INDEX));
    uint32_t M = *(attr->GetAttrPointer<uint32_t>(QUERY_POINT_SIZE_INDEX));
    float radius = *(attr->GetAttrPointer<float>(RADIUS_INDEX));
    float hmin = *(attr->GetAttrPointer<float>(HMIN_INDEX));
    float hmax = *(attr->GetAttrPointer<float>(HMAX_INDEX));
    uint32_t nsample = *(attr->GetAttrPointer<uint32_t>(NSAMPLE_INDEX));

    // 计算单次可处理数据量最大值
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t xyzBlockNum = (N + BLOCK_POINT_SIZE - 1) / BLOCK_POINT_SIZE; // 点云的总块数
    uint32_t tileBlockNum = (ubSize / 2 - 1000 - 8 * nsample) / 340; // 一次最多可以放入的点云的块数
    uint32_t tileDataNum = (tileBlockNum * BLOCK_BYTE_SIZE) / (FLOAT_BYTE_SIZE * 3); // tileBlockNum中对应的点的数量
    // 计算实际输入的分块情况
    uint32_t inputLengthAlign32 = ((N + BLOCK_POINT_SIZE - 1) / BLOCK_POINT_SIZE) * BLOCK_POINT_SIZE * (FLOAT_BYTE_SIZE * 3); // 向上对齐
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    if (aivNum == 0) {
        return ge::GRAPH_FAILED;
    }
    aivNum = (aivNum < B * M) ? aivNum : B * M;
    aivNum = (aivNum >= 1) ? aivNum : 1;

    uint32_t smallTileNum = xyzBlockNum / tileBlockNum;
    uint32_t finalSmallTileNum = (xyzBlockNum % tileBlockNum == 0) ? smallTileNum : smallTileNum + 1; // 遍历点云过程中需要循环的次数
    // 最后一次需要计算的点云点个数
    uint32_t smallTileDataNum = N - (smallTileNum * tileDataNum);
    smallTileDataNum = smallTileDataNum == 0 ? tileDataNum : smallTileDataNum;
    uint32_t smallTileBlockNum = Ceil(smallTileDataNum, BLOCK_POINT_SIZE); // 最后一次循环中参与计算的元素块数
    smallTileBlockNum = (smallTileBlockNum == 0)? tileBlockNum: smallTileBlockNum; // smallTileBlockNum表示的是最后一次循环数据块的数量，而不是简单的取余操作

    uint32_t totalQueryPiont = B * M; // 总查询点的数量
    uint32_t totalTask = B * M; // 总的task数量，每一个task计算八个查询点的圆柱查询
    uint32_t coreTask = Ceil(totalQueryPiont, aivNum); // 平均每个大core的task任务
    uint32_t bigCoreCount = (totalQueryPiont % aivNum == 0)? aivNum : (totalQueryPiont % aivNum);
    uint32_t tailTaskNum = Ceil(totalTask, coreTask); // 尾核的任务数量

    bool dtype = context->GetInputDesc(INPUT_NEW_XYZ_IDX)->GetDataType() == ge::DT_FLOAT;

    context->SetBlockDim(aivNum);

    tiling.set_batchSize(B);
    tiling.set_pointCloudSize(N);
    tiling.set_queryPointSize(M);
    tiling.set_radius(radius);
    tiling.set_hmin(hmin);
    tiling.set_hmax(hmax);
    tiling.set_nsample(nsample);

    tiling.set_coreTask(coreTask);
    tiling.set_tailTaskNum(tailTaskNum);

    tiling.set_bigCoreCount(bigCoreCount);
    tiling.set_finalSmallTileNum(finalSmallTileNum);
    tiling.set_smallTileDataNum(smallTileDataNum);
    tiling.set_tileDataNum(tileDataNum);
    tiling.set_tileBlockNum(tileBlockNum);
    tiling.set_smallTileBlockNum(smallTileBlockNum);

    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(platformInfo);

    // workspace
    uint32_t sysWorkspaceSize = platform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    CHECK_NULLPTR(currentWorkspace);
    currentWorkspace[0] = sysWorkspaceSize;

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class CylinderQuery : public OpDef {
public:
    explicit CylinderQuery(const char* name) : OpDef(name)
    {
        // Tensor输入
        this->Input("new_xyz")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("xyz")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("rot")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("origin_index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        
        // 属性输入
        this->Attr("batch_size")
            .AttrType(REQUIRED)
            .Int();
        this->Attr("point_cloud_size")
            .AttrType(REQUIRED)
            .Int();
        this->Attr("query_point_size")
            .AttrType(REQUIRED)
            .Int();
        this->Attr("radius")
            .AttrType(REQUIRED)
            .Float();
        this->Attr("hmin")
            .AttrType(REQUIRED)
            .Float();
        this->Attr("hmax")
            .AttrType(REQUIRED)
            .Float();
        this->Attr("nsample")
            .AttrType(REQUIRED)
            .Int();

        // Tensor输出
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShapeForCylinderQuery)
            .SetInferDataType(ge::InferDataTypeForCylinderQuery);

        this->AICore().SetTiling(optiling::TilingForCylinderQuery);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(CylinderQuery);
}  // namespace ops