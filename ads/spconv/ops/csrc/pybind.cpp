#include <torch/extension.h>
#include "csrc/pybind.h"
#include "functions.h"

void init_spconv(pybind11::module &m)
{
    // npu_subm_sparse_conv3d
    m.def("npu_subm_sparse_conv3d", &npu_subm_sparse_conv3d);;
}
