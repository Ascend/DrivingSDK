#include "csrc/OpApiCommon.h"
#include "functions.h"
#include <ATen/ATen.h>

std::tuple <at::Tensor, at::Tensor> nms3d(const at::Tensor& boxes,
                                          double threshold)
{
    int32_t box_num = boxes.size(0);
    int32_t data_align = 16;
    int32_t mask_num = ((box_num - 1) / data_align + 1) * data_align;
    at::Tensor mask =
            at::empty({box_num, mask_num}, boxes.options().dtype(at::kShort));
    EXEC_NPU_CMD(aclnnNms3d, boxes, threshold, mask);

    at::Tensor keep = at::zeros({box_num}, mask.options());
    at::Tensor num_out = at::zeros(1, mask.options());
    EXEC_NPU_CMD(aclnnGatherNms3dMask, mask, keep, num_out);
    return std::tie(keep, num_out);
}
