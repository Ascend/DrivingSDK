#include <torch/extension.h>
#include "csrc/pybind.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    init_common(m);
    init_perception_fused(m);
    init_perception_point(m);
    init_perception_vision(m);
}
