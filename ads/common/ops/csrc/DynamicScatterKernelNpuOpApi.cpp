#include <ATen/ATen.h>
#include <torch/csrc/autograd/custom_function.h>
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"
#include "torch_npu/csrc/framework/utils/NpuUtils.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "functions.h"
#include "csrc/common.h"
#include "csrc/OpApiCommon.h"

using npu_preparation = at_npu::native::OpPreparation;
using torch::autograd::Function;
using torch::autograd::AutogradContext;
using tensor_tuple = std::tuple<at::Tensor, at::Tensor, at::Tensor>;

namespace {
inline void npu_dynamic_scatter_check(
    int64_t reduce_type)
{
    TORCH_CHECK(reduce_type == 0 || reduce_type == 1 || reduce_type == 2,
                "reduce_type must be 0(sum) or 1(mean) or 2(max).");
}
} // namespace

static std::map<int64_t, std::string> REDUCE_TYPE_MAP = {{0, "sum"}, {1, "mean"}, {2, "max"}};

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> npu_dynamic_scatter(
    const at::Tensor &feats,
    const at::Tensor &coors,
    int64_t reduce_type)
{
    npu_dynamic_scatter_check(reduce_type);
    auto num_input = feats.size(0);
    auto num_feats = feats.size(1);
    if (num_input == 0) {
        return {feats.clone().detach(), coors.clone().detach(),
                coors.new_empty({0}, at::kInt), coors.new_empty({0}, at::kInt)};
    }

    auto coors_clean = coors.masked_fill(coors.lt(0).any(-1, true), -1);

    at::Tensor out_coors_cpu;
    at::Tensor coors_map_cpu;
    at::Tensor reduce_count_cpu;
    at::Tensor coors_clean_cpu = coors_clean.to("cpu");
    std::tie(out_coors_cpu, coors_map_cpu, reduce_count_cpu) = at::unique_dim(coors_clean_cpu, 0, true, true, true);
    if (out_coors_cpu[0][0].lt(0).item<bool>()) {
        out_coors_cpu = out_coors_cpu.slice(0, 1);
        reduce_count_cpu = reduce_count_cpu.slice(0, 1);
        coors_map_cpu = coors_map_cpu - 1;
    }
    coors_map_cpu = coors_map_cpu.to(at::kInt);
    reduce_count_cpu = reduce_count_cpu.to(at::kInt);
    auto npuDevice = coors.device();
    at::Tensor out_coors = out_coors_cpu.to(npuDevice);
    at::Tensor coors_map = coors_map_cpu.to(npuDevice);
    at::Tensor reduce_count = reduce_count_cpu.to(npuDevice);
    
    auto reduced_feats = at::empty({out_coors.size(0), num_feats}, feats.options());

    const char *reduce_type_string = const_cast<char *>(REDUCE_TYPE_MAP[reduce_type] == "max" ? "max" : "sum");
    EXEC_NPU_CMD(aclnnDynamicScatter, feats, coors_map, reduce_type_string, reduced_feats);
    
    if (reduce_type == 1) {
        reduced_feats /= reduce_count.unsqueeze(-1).to(reduced_feats.dtype());
    }
    
    return {reduced_feats, out_coors, coors_map, reduce_count};
}
