import os

import torch
import torch_npu
import ads_c


def _set_env():
    mx_driving_root = os.path.dirname(os.path.abspath(__file__))
    mx_driving_opp_path = os.path.join(mx_driving_root, "packages", "vendors", "customize")
    ascend_custom_opp_path = os.environ.get("ASCEND_CUSTOM_OPP_PATH")
    ascend_custom_opp_path = (
        mx_driving_opp_path if not ascend_custom_opp_path else mx_driving_opp_path + ";" + ascend_custom_opp_path
    )
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = ascend_custom_opp_path

    mx_driving_op_api_so_path = os.path.join(mx_driving_opp_path, "op_api", "lib", "libcust_opapi.so")
    ads_c._init_op_api_so_path(mx_driving_op_api_so_path)


_set_env()
