#ifndef IOU3D_CPU_H
#define IOU3D_CPU_H

#include <torch/serialize/tensor.h>
#include <vector>

int boxes_iou_bev_cpu(at::Tensor boxes_a_tensor, at::Tensor boxes_b_tensor, at::Tensor ans_iou_tensor);
int boxes_aligned_iou_bev_cpu(at::Tensor boxes_a_tensor, at::Tensor boxes_b_tensor, at::Tensor ans_iou_tensor);
int nms_cpu(at::Tensor boxes, at::Tensor keep, float nms_overlap_thresh);
#endif
