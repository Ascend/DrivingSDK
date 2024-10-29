#include "csrc/pybind.h"
#include <torch/extension.h>

#include <mutex>
#include <string>

std::string g_opApiSoPath;
std::once_flag init_flag; // Flag for one-time initialization

void init_op_api_so_path(const std::string& path)
{
    std::call_once(init_flag, [&]() { g_opApiSoPath = path; });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("_init_op_api_so_path", &init_op_api_so_path);
    init_common(m);
    init_fused(m);
    init_point(m);
    init_preprocess(m);
    init_detection(m);
    init_spconv(m);
}
