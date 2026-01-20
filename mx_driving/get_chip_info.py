import os
import logging
import ctypes
from ctypes import cdll

logging.basicConfig(level=logging.INFO)

def find_library_in_ld_path(lib_name):
    ld_paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    for path in ld_paths:
        if not path.strip():
            continue
        full_path = os.path.join(path.strip(), lib_name)
        if os.path.isfile(full_path):
            return full_path
    return None

class Dsmi_dc_Func:
    def __init__(self, cur=None):
        if cur is None:
            lib_path = find_library_in_ld_path("libdrvdsmi_host.so")
            if lib_path is None:
                ld_lib_path = os.environ.get("LD_LIBRARY_PATH", "")
                logging.info("LD_LIBRARY_PATH: %s", ld_lib_path.split(":"))
                raise FileNotFoundError(
                    "Could not find libdrvdsmi_host.so in LD_LIBRARY_PATH"
                )
            cur = cdll.LoadLibrary(lib_path)
        self.cur = cur

    def chip_version_h(self):
        class Chip(ctypes.Structure):
            _fields_ = [("chip_type", ctypes.c_char * 32),
                        ("chip_name", ctypes.c_char * 32),
                        ("chip_version", ctypes.c_char * 32)
                        ]

        ch = Chip()
        ret = self.cur.dsmi_get_chip_info(0, ctypes.byref(ch))
        chip_name=ch.chip_name.decode(encoding='utf-8')
        return chip_name

def detect_use_arch35():
    dsmi = Dsmi_dc_Func()
    soc_version = dsmi.chip_version_h()
    # 判断是否为A5
    if soc_version and ("95" in soc_version or soc_version == "Ascend950PR"):
        return True
    return False