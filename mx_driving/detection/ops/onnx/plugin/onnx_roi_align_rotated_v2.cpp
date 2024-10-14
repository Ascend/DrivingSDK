/* Copyright (C) Huawei Technologies Co., Ltd 2024. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
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

static const int REQ_ATTR_NUM = 6;

Status ParseParamsRoiAlignRotatedV2(const Message* op_src, ge::Operator &op_dest)
{
    const NodeProto *node = reinterpret_cast<const NodeProto *>(op_src);
    if (node == nullptr) {
        return FAILED;
    }

    bool aligned = true;
    bool clockwise = false;
    int pooled_height = 1;
    int pooled_width = 1;
    int sampling_ratio = 0;
    float spatial_scale = 0.5;

    int required_attr_num = 0;
    for (const auto& attr: node->attribute()) {
        if (attr.name() == "aligned" && attr.type() == ge::onnx::AttributeProto::INT) {
            aligned = attr.i();
            required_attr_num++;
        } else if (attr.name() == "clockwise" && attr.type() == ge::onnx::AttributeProto::INT) {
            clockwise = attr.i();
            required_attr_num++;
        } else if (attr.name() == "pooled_height" && attr.type() == ge::onnx::AttributeProto::INT) {
            pooled_height = attr.i();
            required_attr_num++;
        } else if (attr.name() == "pooled_width" && attr.type() == ge::onnx::AttributeProto::INT) {
            pooled_width = attr.i();
            required_attr_num++;
        } else if (attr.name() == "sampling_ratio" && attr.type() == ge::onnx::AttributeProto::INT) {
            sampling_ratio = attr.i();
            required_attr_num++;
        } else if (attr.name() == "spatial_scale" && attr.type() == ge::onnx::AttributeProto::FLOAT) {
            spatial_scale = attr.f();
            required_attr_num++;
        }
    }

    if (required_attr_num != REQ_ATTR_NUM) {
        return FAILED;
    }

    op_dest.SetAttr("spatial_scale", spatial_scale);
    op_dest.SetAttr("sampling_ratio", sampling_ratio);
    op_dest.SetAttr("pooled_h", pooled_height);
    op_dest.SetAttr("pooled_w", pooled_width);
    op_dest.SetAttr("aligned", aligned);
    op_dest.SetAttr("clockwise", clockwise);

    return SUCCESS;
}

REGISTER_CUSTOM_OP("RoiAlignRotatedV2")
    .FrameworkType(ONNX)
    .OriginOpType({
                    ge::AscendString("mmdeploy::1::RoiAlignRotatedV2"),
                    ge::AscendString("ai.onnx::11::RoiAlignRotatedV2"),
                    ge::AscendString("ai.onnx::12::RoiAlignRotatedV2"),
                    ge::AscendString("ai.onnx::13::RoiAlignRotatedV2"),
                    ge::AscendString("ai.onnx::14::RoiAlignRotatedV2"),
                    ge::AscendString("ai.onnx::15::RoiAlignRotatedV2"),
                    ge::AscendString("ai.onnx::16::RoiAlignRotatedV2"),
                    ge::AscendString("ai.onnx::17::RoiAlignRotatedV2"),
                    ge::AscendString("ai.onnx::18::RoiAlignRotatedV2")})
    .ParseParamsFn(ParseParamsRoiAlignRotatedV2)
    .ImplyType(ImplyType::TVM);
}