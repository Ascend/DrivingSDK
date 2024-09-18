#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <vector>

#include "iou3d_cpu.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("boxes_aligned_iou_bev_cpu", &boxes_aligned_iou_bev_cpu, "aligned oriented boxes iou");
	m.def("boxes_iou_bev_cpu", &boxes_iou_bev_cpu, "oriented boxes iou");
	m.def("nms_cpu", &nms_cpu, "nms cpu");
}
