#include <torch/extension.h>
#include "../ads/common/ops/csrc/functions.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    init_common(m);
}
