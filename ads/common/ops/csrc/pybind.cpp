#include <torch/extension.h>
#include "functions.h"

void init_common(pybind11::module &m)
{
    m.def("npu_scatter_max", &npu_scatter_max);
    m.def("npu_scatter_max_backward", &npu_scatter_max_backward);

    // rotatedBox kernel
    m.def("npu_rotated_box_decode", &npu_rotated_box_decode, "npu_rot_box_decode NPU version");
    m.def("npu_rotated_box_encode", &npu_rotated_box_encode, "npu_rot_box_encode NPU version");

    // rotated iou
    m.def("npu_rotated_iou", &npu_rotated_iou, "npu_rotated_iou NPU version");

    // roated overlap
    m.def("npu_rotated_overlaps", &npu_rotated_overlaps, "npu_rotated_overlap NPU version");

    // sign bits
    m.def("npu_sign_bits_pack", &npu_sign_bits_pack, "npu_sign_bits_pack NPU version");
    m.def("npu_sign_bits_unpack", &npu_sign_bits_unpack, "npu_sign_bits_unpack NPU version");

    // softmax
    m.def("npu_softmax_cross_entropy_with_logits", &npu_softmax_cross_entropy_with_logits, "npu_softmax_cross_entropy_with_logits NPU version");

    // stride add
    m.def("npu_stride_add", &npu_stride_add, "npu_stride_add NPU version");

    // transpose
    m.def("npu_transpose", &npu_transpose, "npu_transpose NPU version");

    // yolo encode
    m.def("npu_yolo_boxes_encode", &npu_yolo_boxes_encode, "npu_yolo_boxes_encode NPU version");

    // scatter
    m.def("npu_scatter", &npu_scatter, "npu_scatter NPU version");

    // silu
    m.def("npu_silu_", &npu_silu_);
    m.def("npu_silu", &npu_silu);

    // rotary mul
    m.def("npu_rotary_mul", &npu_rotary_mul);

    m.def("npu_abs", &npu_abs);

    // npu_fast_gelu
    m.def("npu_fast_gelu", &npu_fast_gelu);
    m.def("npu_fast_gelu_backward", &npu_fast_gelu_backward);

    // npu_anchor_response_flags
    m.def("npu_anchor_response_flags", &npu_anchor_response_flags);

    // npu_bounding_box_decode
    m.def("npu_bounding_box_decode", &npu_bounding_box_decode);

    // npu_bounding_box_encode
    m.def("npu_bounding_box_encode", &npu_bounding_box_encode);

    // npu_batch_nms
    m.def("npu_batch_nms", &npu_batch_nms);

    // npu_confusion_transpose
    m.def("npu_confusion_transpose", &npu_confusion_transpose);
    m.def("npu_confusion_transpose_backward", &npu_confusion_transpose_backward);

    // npu_broadcast
    m.def("npu_broadcast", &npu_broadcast);
    
    // npu_moe_tutel
    m.def("npu_moe_tutel", &npu_moe_tutel, "npu_moe_tutel NPU version");
    m.def("npu_moe_tutel_data_backward", &npu_moe_tutel_data_backward, "npu_moe_tutel_data_backward NPU version");
    m.def("npu_moe_tutel_gate_backward", &npu_moe_tutel_gate_backward, "npu_moe_tutel_gate_backward NPU version");
}
