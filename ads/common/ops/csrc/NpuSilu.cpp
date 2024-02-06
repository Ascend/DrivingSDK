#include "torch_npu/csrc/framework/OpCommand.h"
#include "functions.h"
#include "common.h"

at::Tensor &silu_out_npu_nocheck(at::Tensor &result, const at::Tensor &self)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("Swish")
        .Input(self)
        .Output(result)
        .Attr("scale", (float)1.0)
        .Run();
    return result;
}

at::Tensor &silu_out_npu(const at::Tensor &self, at::Tensor &result)
{
    if (!check_match(result)) {
        at::Tensor contiguous_result = result.contiguous();
        silu_out_npu_nocheck(contiguous_result, self);
        format_fresh_view(result, contiguous_result);
    } else {
        silu_out_npu_nocheck(result, self);
    }

    return result;
}

at::Tensor silu_kernel_npu(const at::Tensor &self)
{
    at::Tensor result = at::empty(self.sizes(), self.options());

    silu_out_npu_nocheck(result, self);

    return result;
}

at::Tensor &silu_backward_out_npu_nocheck(
    at::Tensor &result,
    const at::Tensor &grad_output,
    const at::Tensor &x0,
    const at::Tensor &x1)
{
    at_npu::native::OpCommand cmd;
    cmd.Name("SwishGrad")
        .Input(grad_output)
        .Input(x0)
        .Input(x1)
        .Output(result)
        .Run();

    return result;
}

at::Tensor npu_silu_backward(const at::Tensor &grad_output, const at::Tensor &x0, const at::Tensor &x1)
{
    at::Tensor grad_input = at::empty(grad_output.sizes(), grad_output.options());
    silu_backward_out_npu_nocheck(grad_input, grad_output, x0, x1);

    return grad_input;
}

at::Tensor npu_silu(const at::Tensor &self)
{
    return silu_kernel_npu(self);
}

at::Tensor &npu_silu_(at::Tensor &self)
{
    silu_out_npu(self, self);
    return self;
}
