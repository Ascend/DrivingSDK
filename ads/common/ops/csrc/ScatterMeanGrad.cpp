#include <ATen/ATen.h>
#include "csrc/OpApiCommon.h"
#include "functions.h"

at::Tensor npu_scatter_mean_grad(const at::Tensor& grad_out, const at::Tensor& index, int32_t dim)
{
    // construct the output tensor of the NPU
    auto index_size = index.sizes();
    auto grad_out_size = grad_out.sizes();
    auto index_dims = index.sizes().size();
    auto grad_out_dims = grad_out_size.size();
    TORCH_CHECK(grad_out.scalar_type() == at::kFloat,
        "grad_out: float32 tensor expected but got a tensor with dtype: ", grad_out.scalar_type());
    TORCH_CHECK(index.scalar_type() == at::kInt,
        "index: int32 tensor expected but got a tensor with dtype: ",
        index.scalar_type());
    TORCH_CHECK(grad_out_dims != 0 && index_dims != 0,
        "grad_out and index should not be empty");
    TORCH_CHECK(grad_out_dims == index_dims || dim == index_dims - 1 || dim == -1,
        "the dims of grad_out and index should be the same");

    c10::SmallVector<int64_t, 8> grad_in_size;
    for (int32_t i = 0; i < grad_out_dims; i++) {
        grad_in_size.push_back(grad_out_size[i]);
    }
    dim = (dim + index_dims) % index_dims;
    grad_in_size[dim] = index_size[dim];
    for (int32_t i = 0; i < grad_out_dims; i++) {
        TORCH_CHECK(i >= index_dims || grad_in_size[i] == index_size[i],
            "the shape except dim should be the same");
    }
    
    at::Tensor result = at::empty(grad_in_size, grad_out.options());
    EXEC_NPU_CMD(aclnnScatterMeanGrad, grad_out, index, dim, result);
    return result;
}