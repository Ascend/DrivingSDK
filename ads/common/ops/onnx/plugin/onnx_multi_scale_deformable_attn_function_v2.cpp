/* Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "graph/operator.h"
#include "register/register.h"
#include "proto/onnx/ge_onnx.pb.h"

using namespace ge;

namespace domi {
using NodeProto = ge::onnx::NodeProto;

Status ParseOnnxParamsMultiScaleDeformableAttnFunctionV2(const Message *op_src, ge::Operator &op_dest)
{
    // trans op_src to op_dest
    // if op_src get required attr failed, need to return Failed
    // if op_src get optional attr failed, need to return Failed or set a default value

    const NodeProto *node = reinterpret_cast<const NodeProto *>(op_src);
    if (node == nullptr) {
        return FAILED;
    }

    return SUCCESS;
}

REGISTER_CUSTOM_OP("MultiScaleDeformableAttnFunctionV2")
    .FrameworkType(ONNX)
    .OriginOpType({
                    ge::AscendString("npu::1::MultiScaleDeformableAttnFunctionV2"),
                    ge::AscendString("ai.onnx::8::MultiScaleDeformableAttnFunctionV2"),
                    ge::AscendString("ai.onnx::9::MultiScaleDeformableAttnFunctionV2"),
                    ge::AscendString("ai.onnx::10::MultiScaleDeformableAttnFunctionV2"),
                    ge::AscendString("ai.onnx::11::MultiScaleDeformableAttnFunctionV2"),
                    ge::AscendString("ai.onnx::12::MultiScaleDeformableAttnFunctionV2"),
                    ge::AscendString("ai.onnx::13::MultiScaleDeformableAttnFunctionV2")})
    .ParseParamsFn(ParseOnnxParamsMultiScaleDeformableAttnFunctionV2)
    .ImplyType(ImplyType::TVM);
} // domi

