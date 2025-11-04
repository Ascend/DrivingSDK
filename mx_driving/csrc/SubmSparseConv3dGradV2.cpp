#include "csrc/OpApiCommon.h"
#include "csrc/functions.h"


std::tuple<at::Tensor, at::Tensor> npu_subm_sparse_conv3d_grad_v2(
    const at::Tensor& features, 
    const at::Tensor& weight,
    const at::Tensor& grad_out_features,
    const at::Tensor& indices_offset
    )
{
    TORCH_CHECK_NPU(features);
    TORCH_CHECK_NPU(weight);
    TORCH_CHECK_NPU(grad_out_features);
    TORCH_CHECK_NPU(indices_offset);

    auto features_size = features.sizes();
    auto weight_size = weight.sizes();

    at::Tensor features_grad = at::zeros(features_size, features.options());
    at::Tensor weight_grad = at::zeros(weight_size, weight.options());

    // zero init
    if (features.options().dtype() == at::kFloat) {

        EXEC_NPU_CMD(aclnnSubmSparseConv3dGradV2, features, weight, grad_out_features, indices_offset, 
            features_grad, weight_grad);

    } else {
        at::Tensor weight_grad_fp32 = at::zeros(weight_size, weight.options().dtype(at::kFloat));

        EXEC_NPU_CMD(aclnnSubmSparseConv3dGradV2, features, weight, grad_out_features, indices_offset, 
            features_grad, weight_grad_fp32);

        weight_grad = weight_grad_fp32.to(at::kHalf);
    }
    
    return std::tie(features_grad, weight_grad);
}