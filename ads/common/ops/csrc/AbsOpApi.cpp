#include <ATen/ATen.h>
#include "OpApiCommon.h"
#include "functions.h"

at::Tensor npu_abs(const at::Tensor& self)
{
    // construct the output tensor of the NPU
    at::Tensor result = at::empty(self.sizes(), self.options());

    // calculate the output result of the NPU
    EXEC_NPU_CMD(aclnnAbs, self, result);
    return result;
}
