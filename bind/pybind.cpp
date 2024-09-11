#include <torch/extension.h>
#include "csrc/pybind.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    init_common(m);
    init_fused(m);
    init_point(m);
    init_preprocess(m);
    init_detection(m);
    init_spconv(m);
}
