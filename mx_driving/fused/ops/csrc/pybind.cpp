#include "csrc/pybind.h"

#include <torch/extension.h>

#include "functions.h"
void init_fused(pybind11::module& m)
{
    // nnpu_max_pool2d
    m.def("npu_max_pool2d", &npu_max_pool2d);
    // npu_multi_scale_deformable_attn_function
    m.def("npu_multi_scale_deformable_attn_function", &npu_multi_scale_deformable_attn_function);
    m.def("multi_scale_deformable_attn_grad", &multi_scale_deformable_attn_grad);
    m.def("multi_scale_deformable_attn_grad_v2", &multi_scale_deformable_attn_grad_v2);

    // npu_add_relu
    m.def("npu_add_relu", &npu_add_relu);
    m.def("npu_add_relu_grad", &npu_add_relu_grad);

    // fused_bias_leaky_relu
    m.def("fused_bias_leaky_relu", &fused_bias_leaky_relu);
    
    // npu_deformable_aggregation
    m.def("npu_deformable_aggregation", &deformable_aggregation);
    m.def("npu_deformable_aggregation_grad", &deformable_aggregation_grad);

    // deformable_conv2d
    m.def("npu_deformable_conv2d", &npu_deformable_conv2d);
}
