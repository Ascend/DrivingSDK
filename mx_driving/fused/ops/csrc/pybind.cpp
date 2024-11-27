#include "csrc/pybind.h"

#include <torch/extension.h>

#include "functions.h"
void init_fused(pybind11::module& m)
{
    // nnpu_max_pool2d
    m.def("npu_max_pool2d", &npu_max_pool2d);
    // mullti_scale_deformable_attn
    m.def("multi_scale_deformable_attn", &multi_scale_deformable_attn);
    m.def("multi_scale_deformable_attn_backward", &multi_scale_deformable_attn_backward);

    // npu_add_relu
    m.def("npu_add_relu", &npu_add_relu);
    m.def("npu_add_relu_grad", &npu_add_relu_grad);

    // fused_bias_leaky_relu
    m.def("fused_bias_leaky_relu", &fused_bias_leaky_relu);

    // npu_deformable_aggregation
    m.def("npu_deformable_aggregation", &deformable_aggregation);
    m.def("npu_deformable_aggregation_grad", &deformable_aggregation_grad);

    // deformable_conv2d
    m.def("deformable_conv2d", &deformable_conv2d);
    m.def("modulated_deformable_conv2d", &modulated_deformable_conv2d);
    m.def("deformable_conv2d_backward", &deformable_conv2d_backward);
    m.def("modulated_deformable_conv2d_backward", &modulated_deformable_conv2d_backward);

    // npu_geometric_kernel_attention_func
    m.def("npu_geometric_kernel_attention_func", &npu_geometric_kernel_attention_func);
    m.def("npu_geometric_kernel_attention_backward", &npu_geometric_kernel_attention_backward);
}
