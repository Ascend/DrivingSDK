import os
import shutil
import unittest
import torch
import torch_npu
import torch_npu.onnx

from torch_npu.utils.path_manager import PathManager
from torch_npu.testing.testcase import run_tests


import ads.common.ops.onnx as onnx_ads



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

    def test_addcustom_export_onnx(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x, y):
                return onnx_ads.my_add(x, y)

        def export_onnx(onnx_name):
            x = torch.rand([8, 2048]).half().npu()
            y = torch.rand([8, 2048]).half().npu()
            model = Model().to("npu")
            model(x, y)
            self.onnx_export(model, (x, y), onnx_name, ["input1", "input2"], ["out"])
        
        onnx_name = "model_ads_add_custom.onnx"
        export_onnx(onnx_name)
        if not os.path.isfile(os.path.join(OpsOnnx.onnx_dir, onnx_name)):
            raise FileNotFoundError("No such file:", onnx_name)
  
if __name__ == '__main__':
    run_tests()
