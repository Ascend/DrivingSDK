#include <torch/extension.h>
#include "csrc/pybind.h"
#include "functions.h"

void init_detection(pybind11::module& m)
{
    // nms3d_normal
    m.def("nms3d_normal", &nms3d_normal);

    // nms3d
    m.def("nms3d", &nms3d);

    // roated overlap
    m.def("npu_rotated_overlaps", &npu_rotated_overlaps, "npu_rotated_overlap NPU version");

    // rotated iou
    m.def("npu_rotated_iou", &npu_rotated_iou);
    
    // npu_boxes_overlap_bev
    m.def("npu_boxes_overlap_bev", &npu_boxes_overlap_bev, "boxes_overlap_bev NPU version");

    // roi_align_rotated_v2_forward_npu
    m.def("roi_align_rotated_v2_forward_npu", &roi_align_rotated_v2_forward_npu);

    // npu_roi_align_rotated_grad_v2
    m.def("npu_roi_align_rotated_grad_v2", &npu_roi_align_rotated_grad_v2);

    // npu_box_iou_quadri
    m.def("npu_box_iou_quadri", &npu_box_iou_quadri, "box_iou_quadri NPU version");

    // npu_box_iou_rotated
    m.def("npu_box_iou_rotated", &npu_box_iou_rotated, "box_iou_rotated NPU version");

    // border_align_forward_npu
    m.def("border_align_forward_npu", &border_align_forward_npu);

    // border_align_backward_npu
    m.def("border_align_backward", &border_align_backward);

    // npu_roiaware_pool3d_forward
    m.def("npu_roiaware_pool3d_forward", &npu_roiaware_pool3d_forward);

    // roiaware_pool3d_grad
    m.def("roiaware_pool3d_grad", &roiaware_pool3d_grad, "roiaware_pool3d_grad NPU version");

    // pixel_group
    m.def("pixel_group", &pixel_group, "pixel_group");
}
