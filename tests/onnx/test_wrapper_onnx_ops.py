import os
import shutil
import unittest
import torch
import torch_npu
import torch_npu.onnx

from torch_npu.utils.path_manager import PathManager
from torch_npu.testing.testcase import run_tests
import mx_driving.fused.ops.onnx as onnx_ads


DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class OpsOnnx(torch_npu.testing.testcase.TestCase):

    current_directory = os.path.realpath(os.path.dirname(__file__))
    onnx_dir = os.path.join(current_directory, "test_onnx_wrapper_ops")

    @classmethod
    def setUpClass(cls):
        PathManager.make_dir_safety(OpsOnnx.onnx_dir)

    @classmethod
    def tearDownClass(cls):
        if not os.path.exists(OpsOnnx.onnx_dir):
            raise FileNotFoundError("No such directory:", OpsOnnx.onnx_dir)
        PathManager.remove_path_safety(OpsOnnx.onnx_dir)

    def onnx_export(self, model, inputs, onnx_name, inputnames=None, outputnames=None):
        if inputnames is None:
            inputnames = ["inputnames"]
        if outputnames is None:
            outputnames = ["outputnames"]
        model.eval()
        OPS_EXPORT_TYPE = torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK

        with torch.no_grad():
            torch.onnx.export(model, inputs,
                              os.path.join(OpsOnnx.onnx_dir, onnx_name),
                              opset_version=11,
                              operator_export_type=OPS_EXPORT_TYPE,
                              input_names=inputnames,
                              output_names=outputnames)
        
    @unittest.skipIf(DEVICE_NAME != 'Ascend910B', "OP `MultiScaleDeformableAttnFunction` is only supported on 910B, skip this ut!")
    def test_msda_export_onnx(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, value, shapes, level_start_index, sampling_locations, attention_weights):
                return onnx_ads.onnx_msda(value, shapes, level_start_index, sampling_locations, attention_weights)

        def export_onnx(onnx_name):
            value = torch.rand([6, 114400, 8, 32]).to(torch.float32).npu()
            shapes = torch.tensor([[160, 140],
                                   [130, 120],
                                   [160, 240],
                                   [130, 120],
                                   [160, 140]]).to(torch.int32).npu()
            level_start_index = torch.tensor([0, 22400, 38000, 76400, 92000]).to(torch.int32).npu()
            sampling_locations = torch.rand([6, 10000, 8, 5, 4, 2]).to(torch.float32).npu()
            attention_weights = torch.rand([6, 10000, 8, 5, 4]).to(torch.float32).npu()

            model = Model().to("npu")
            model(value, shapes, level_start_index,
                  sampling_locations, attention_weights)
            self.onnx_export(model,
                             (value, shapes, level_start_index, sampling_locations, attention_weights),
                             onnx_name,
                             ["value", "shapes", "level_start_index", "sampling_locations", "attention_weights"],
                             ["out"])

        onnx_name = "model_ads_msda.onnx"
        export_onnx(onnx_name)
        if not os.path.isfile(os.path.join(OpsOnnx.onnx_dir, onnx_name)):
            raise FileNotFoundError("No such file:", onnx_name)

if __name__ == '__main__':
    run_tests()
