import os
import logging
import ctypes
from ctypes import cdll

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

def log_driver_lib_files():
    driver_path = "/usr/local/Ascend/driver/lib64/driver"
    try:
        if os.path.exists(driver_path):
            files = os.listdir(driver_path)
            files.sort()  # 可选：排序便于阅读
            logger.info("Files in %s: %s", driver_path, files)
        else:
            logger.warning("Path does not exist: %s", driver_path)
    except OSError as e:
        logger.error("Failed to list files in %s: %s", driver_path, e)

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
                log_driver_lib_files()
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