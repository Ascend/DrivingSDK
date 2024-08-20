#include <torch/extension.h>
#include "csrc/pybind.h"
#include "functions.h"

void init_spconv(pybind11::module &m)
{
    // npu_subm_sparse_conv3d
    m.def("npu_subm_sparse_conv3d", &npu_subm_sparse_conv3d);

    // npu_sparse_conv3d
    m.def("npu_sparse_conv3d", &npu_sparse_conv3d);

    // npu_sparse_inverse_conv3d
    m.def("npu_sparse_inverse_conv3d", &npu_sparse_inverse_conv3d);

    // multi_to_sparse
    m.def("multi_to_sparse", &multi_to_sparse);

    // npu_sparse_conv3d_grad
    m.def("npu_sparse_conv3d_grad", &npu_sparse_conv3d_grad);
}
