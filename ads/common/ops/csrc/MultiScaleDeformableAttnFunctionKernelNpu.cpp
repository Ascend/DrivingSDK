#include <ATen/ATen.h>
#include "OpApiCommon.h"
#include "functions.h"

at::Tensor npu_multi_scale_deformable_attn_function(const at::Tensor& value, const at::Tensor& shape, const at::Tensor& offset,
                                                    const at::Tensor& location, const at::Tensor& weight)
{
    // construct the output tensor of the NPU
    auto value_size = value.sizes();
    auto location_size = location.sizes();
    auto output_size = {value_size[0], location_size[1], value_size[2] * value_size[3]};
    at::Tensor result = at::empty(output_size, value.options());
    // at::Tensor result = at::empty(self.sizes(), value.options());

    // calculate the output result of the NPU
    EXEC_NPU_CMD(aclnnMultiScaleDeformableAttnFunction, value, shape, offset, location, weight, result);
    return result;
}
