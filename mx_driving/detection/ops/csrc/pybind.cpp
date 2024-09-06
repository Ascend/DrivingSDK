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
}
