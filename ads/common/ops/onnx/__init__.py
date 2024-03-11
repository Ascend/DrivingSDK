from .wrapper_onnx_ops import NPUAddCustomOP
from .wrapper_onnx_ops import NPUMultiScaleDeformableAttnFunctionV2OP


my_add = NPUAddCustomOP.apply
onnx_msda = NPUMultiScaleDeformableAttnFunctionV2OP.apply