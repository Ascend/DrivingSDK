
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 */
#include "sigmoid_focal_loss_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "ge/utils.h"

namespace optiling {
    constexpr uint32_t BLOCK_SIZE = 32;
    constexpr float AVAILABLE_UB_RATIO = 0.6;
}

namespace optiling {
    static ge::graphStatus TilingForSigmoidFocalLossGrad(gert::TilingContext *context) {
        CHECK_NULLPTR(context);
        auto logitShapePtr = context->GetInputShape(0); // [N, C]
        auto attrPtr = context->GetAttrs();
        auto platformInfoPtr = context->GetPlatformInfo();
        
        CHECK_NULLPTR(logitShapePtr);
        CHECK_NULLPTR(attrPtr);
        CHECK_NULLPTR(platformInfoPtr);
        
        auto platformInfo = platform_ascendc::PlatformAscendC(platformInfoPtr);
        uint32_t numSamples = logitShapePtr->GetStorageShape().GetDim(0);
        uint32_t numClasses = logitShapePtr->GetStorageShape().GetDim(1);
        if (numClasses == 0) {
            return ge::GRAPH_FAILED;
        }

        auto gammaPtr = attrPtr->GetAttrPointer<float>(0);
        auto alphaPtr = attrPtr->GetAttrPointer<float>(1);
        CHECK_NULLPTR(gammaPtr);
        CHECK_NULLPTR(alphaPtr);
        float gamma = *gammaPtr;
        float alpha = *alphaPtr;

        auto logitDataTypePtr = context->GetInputDesc(0);
        CHECK_NULLPTR(logitDataTypePtr);
        auto logitDataType = logitDataTypePtr->GetDataType();
        uint32_t byteSizePerElement = (logitDataType == ge::DT_FLOAT16) ? 2 : 4;
        
        uint32_t coreNum = platformInfo.GetCoreNumAiv();
        if (coreNum == 0) {
            return ge::GRAPH_FAILED;
        }
        uint32_t usedCoreNum = coreNum;

        // 按numSamples分核
        uint32_t numHeadCores;
        uint32_t numTailCores; 
        uint32_t numTaskOnHeadCore;
        uint32_t numTaskOnTailCore;
        uint32_t totalTaskCount = numSamples;
        uint32_t numTaskTmp = totalTaskCount / coreNum;
        if (numTaskTmp * coreNum == totalTaskCount) {
            numHeadCores = coreNum;                        // 大核数量
            numTailCores = 0;                              // 小核数量
            numTaskOnHeadCore = numTaskTmp;                // 每个大核上分配的任务数
            numTaskOnTailCore = 0;                         // 每个小核上分配的任务数
        } else {
            numHeadCores = totalTaskCount - numTaskTmp * coreNum;         // 大核数量
            numTailCores = coreNum - numHeadCores;          // 小核数量
            numTaskOnHeadCore = numTaskTmp + 1;             // 每个大核上分配的任务数
            numTaskOnTailCore = numTaskTmp;                 // 每个小核上分配的任务数
        }
        uint32_t numClassesAlign = AlignUp(numClasses, BLOCK_SIZE / byteSizePerElement);  // 每个task 32byte对齐后的类别数量

        if (numTaskOnTailCore == 0) {
            usedCoreNum = numHeadCores;
        }

        // 设置ub大小
        uint64_t ubSize;
        platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        ubSize *= AVAILABLE_UB_RATIO;
        
        // 计算大核上每次搬运的任务数，搬运次数，尾任务数
        uint32_t numLoopOnHeadCore = 0;
        uint32_t numTaskPerLoopOnHeadCore = std::min(numTaskOnHeadCore, uint32_t(ubSize / (numClassesAlign * byteSizePerElement * 8)));     // 按分配的UB空间每次能搬运的最多的task数
        if (numTaskPerLoopOnHeadCore != 0) {
            numLoopOnHeadCore = numTaskOnHeadCore / numTaskPerLoopOnHeadCore;                          // 大核上搬运次数
        }
        uint32_t numTaskTailOnHeadCore = numTaskOnHeadCore - numTaskPerLoopOnHeadCore * numLoopOnHeadCore;  // 大核上尾任务数

        // 计算小核上每次搬运的任务数，搬运次数，尾任务数
        uint32_t numLoopOnTailCore = 0;
        uint32_t numTaskPerLoopOnTailCore = std::min(numTaskOnTailCore, uint32_t(ubSize / (numClassesAlign * byteSizePerElement * 8)));    // 小核上常规搬运次数
        if (numTaskPerLoopOnTailCore != 0) {
            numLoopOnTailCore = numTaskOnTailCore / numTaskPerLoopOnTailCore;    // 小核上每次常规搬运的任务数
        }
        uint32_t numTaskTailOnTailCore = numTaskOnTailCore - numTaskPerLoopOnTailCore * numLoopOnTailCore;  // 小核上尾任务数

        // 设置workspace大小
        auto currentWorkspace = context->GetWorkspaceSizes(1);
        CHECK_NULLPTR(currentWorkspace);
        currentWorkspace[0] = 0;

        // 设置tiling参数
        SigmoidFocalLossTilingData TilingData;
        TilingData.set_numSamples(numSamples);
        TilingData.set_numClasses(numClasses);
        TilingData.set_numClassesAlign(numClassesAlign);
        TilingData.set_usedCoreNum(usedCoreNum);
        TilingData.set_numHeadCores(numHeadCores);
        TilingData.set_numTailCores(numTailCores);
        TilingData.set_numTaskOnHeadCore(numTaskOnHeadCore);
        TilingData.set_numTaskOnTailCore(numTaskOnTailCore);
        TilingData.set_numLoopOnHeadCore(numLoopOnHeadCore);
        TilingData.set_numTaskPerLoopOnHeadCore(numTaskPerLoopOnHeadCore);
        TilingData.set_numTaskTailOnHeadCore(numTaskTailOnHeadCore);
        TilingData.set_numLoopOnTailCore(numLoopOnTailCore);
        TilingData.set_numTaskPerLoopOnTailCore(numTaskPerLoopOnTailCore);
        TilingData.set_numTaskTailOnTailCore(numTaskTailOnTailCore);
        TilingData.set_gamma(gamma);
        TilingData.set_alpha(alpha);

        context->SetBlockDim(usedCoreNum);
        CHECK_NULLPTR(context->GetRawTilingData());
        TilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(TilingData.GetDataSize());
        return ge::GRAPH_SUCCESS;

    };
}

namespace ge {
    static ge::graphStatus InferShapeForSigmoidFocalLossGrad(gert::InferShapeContext* context) {
        CHECK_NULLPTR(context);
        const gert::Shape* logit_shape = context->GetInputShape(0);
        gert::Shape* loss_shape = context->GetOutputShape(0);
        if (logit_shape == nullptr || loss_shape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        *loss_shape = *logit_shape;
        return ge::GRAPH_SUCCESS;
    }

    static ge::graphStatus InferDataTypeForSigmoidFocalLossGrad(gert::InferDataTypeContext *context) {
        CHECK_NULLPTR(context);
        const ge::DataType logit_dtype = context->GetInputDataType(0);
        context->SetOutputDataType(0, logit_dtype);
        return ge::GRAPH_SUCCESS;
    }
}

namespace ops {
    class SigmoidFocalLossGrad: public OpDef {
        public:
        explicit SigmoidFocalLossGrad(const char* name) : OpDef(name) {
            this->Input("logit")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
                .AutoContiguous();
            this->Input("target_y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
                .AutoContiguous();
            this->Input("weight_y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
                .AutoContiguous();

            this->Attr("gamma")
                .AttrType(REQUIRED)
                .Float();
            this->Attr("alpha")
                .AttrType(REQUIRED)
                .Float();

            this->Output("grad_input")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        
            this->SetInferShape(ge::InferShapeForSigmoidFocalLossGrad)
                .SetInferDataType(ge::InferDataTypeForSigmoidFocalLossGrad);

            this->AICore().SetTiling(optiling::TilingForSigmoidFocalLossGrad);
            this->AICore().AddConfig("ascend910_95");
        }
    };

    OP_ADD(SigmoidFocalLossGrad);
}

