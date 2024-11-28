import os
import onnx_plugin
from onnx_plugin import helper, TensorProto


def roi_align_rotated():
    feature_map = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 48, 5, 5])
    rois = helper.make_tensor_value_info('rois', TensorProto.FLOAT, [3, 6])

    output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [3, 48, 2, 2])

    node_def = onnx_plugin.helper.make_node('RoiAlignRotatedV2',
                                      inputs=['input', 'rois'],
                                      outputs=['output'],
                                      spatial_scale=1.0,
                                      sampling_ratio=0,
                                      pooled_height=2,
                                      pooled_width=2,
                                      aligned=True,
                                      clockwise=False
    )

    graph_def = helper.make_graph(
        [node_def],
        'test_model',
        [feature_map, rois],
        [output]
    )

    model_def = helper.make_model(graph_def)
    model_def.opset_import[0].version = 11
    current_path = os.path.abspath(__file__)
    idx = current_path.rfind('/')
    current_path = current_path[:idx]
    onnx_plugin.save(model_def, os.path.join(current_path, "roi_align_rotated.onnx"))

if __name__ == "__main__":
    roi_align_rotated()