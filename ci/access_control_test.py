# Copyright (c) 2020 Huawei Technologies Co., Ltd
# All rights reserved.
#
# Licensed under the BSD 3-Clause License  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -*- coding: UTF-8 -*-

import os
import queue
import re
import subprocess
import sys
import threading
import warnings
from abc import ABCMeta, abstractmethod
from pathlib import Path

NUM_DEVICE = 8
BASE_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # project root
TEST_DIR = os.path.join(BASE_DIR, "tests", "torch")


def check_path_owner_consistent(path: str):
    """
    Function Description:
        check whether the path belong to process owner
    Parameter:
        path: the path to check
    Exception Description:
        when invalid path, prompt the user
    """
    if not os.path.exists(path):
        msg = f"The path does not exist: {path}"
        raise RuntimeError(msg)
    if os.stat(path).st_uid != os.getuid():
        warnings.warn(f"Warning: The {path} owner does not match the current user.")


def check_directory_path_readable(path):
    check_path_owner_consistent(path)
    if os.path.islink(path):
        msg = f"Invalid path is a soft chain: {path}"
        raise RuntimeError(msg)
    if not os.access(path, os.R_OK):
        msg = f"The path permission check failed: {path}"
        raise RuntimeError(msg)


class AccurateTest(metaclass=ABCMeta):
    @abstractmethod
    def identify(self, modify_file):
        raise Exception("abstract method. Subclasses should implement it.")

    @staticmethod
    def find_ut_by_regex(regex):
        ut_files = []
        cmd = "find {} -name {}".format(str(TEST_DIR), regex)
        status, output = subprocess.getstatusoutput(cmd)
        if status or not output:
            pass  # 对于找不到的暂时不作处理
        else:
            files = output.split("\n")
            for ut_file in files:
                if ut_file.endswith("run_test.py"):
                    continue
                if ut_file.endswith(".py"):
                    ut_files.append(ut_file)
        return ut_files


class OpStrategy(AccurateTest):
    def split_string(self, filename):
        words = []
        word = ""

        for char in filename:
            if char.isupper():
                if word:
                    words.append(word.lower())
                word = char
            elif char == "_":
                if word:
                    words.append(word.lower())
                word = ""
            else:
                word += char

        if word:
            words.append(word.lower())
        return words

    def identify(self, modify_file):
        filename = Path(modify_file).name
        features = self.split_string(filename)
        # the last word could be v2, tiling, backward etc.
        if len(features) > 1:
            features = features[:-1]
        regex = "*" + "*".join([f"{feature.lower()}" for feature in features]) + "*"
        return AccurateTest.find_ut_by_regex(regex)


class DirectoryStrategy(AccurateTest):
    def identify(self, modify_file):
        path_modify_file = Path(modify_file)
        is_test_file = (
            str(path_modify_file.parts[0]) == "tests"
            and str(path_modify_file.parts[1]) == "torch"
            and re.match("test_(.+).py", Path(modify_file).name)
        )
        return [str(os.path.join(BASE_DIR, modify_file))] if is_test_file else []


class TestMgr:
    def __init__(self):
        self.modify_files = []
        self.test_files = {"ut_files": [], "op_ut_files": []}

    def load(self, modify_files):
        check_directory_path_readable(modify_files)
        with open(modify_files) as f:
            for line in f:
                line = line.strip()
                self.modify_files.append(line)

    def analyze(self):
        for modify_file in self.modify_files:
            self.test_files["ut_files"] += DirectoryStrategy().identify(modify_file)
            self.test_files["ut_files"] += OpStrategy().identify(modify_file)
        unique_files = sorted(set(self.test_files["ut_files"]))

        exist_ut_file = [changed_file for changed_file in unique_files if Path(changed_file).exists()]
        self.test_files["ut_files"] = exist_ut_file

    def get_test_files(self):
        return self.test_files

    def print_modify_files(self):
        print("modify files:")
        for modify_file in self.modify_files:
            print(modify_file)

    def print_ut_files(self):
        print("ut files:")
        for ut_file in self.test_files["ut_files"]:
            print(ut_file)

    def print_op_ut_files(self):
        print("op ut files:")
        for op_ut_file in self.test_files["op_ut_files"]:
            print(op_ut_file)


def exec_ut(files):
    def get_op_name(ut_file):
        return ut_file.split("/")[-1].split(".")[0].lstrip("test_")

    def get_ut_name(ut_file):
        return str(Path(ut_file).relative_to(TEST_DIR))[:-3]

    def get_ut_cmd(ut_type, ut_file):
        cmd = [sys.executable, "run_test.py", "-v", "-i"]
        if ut_type == "op_ut_files":
            return cmd + ["test_ops", "--", "-k", get_op_name(ut_file)]
        return cmd + [get_ut_name(ut_file)]

    def wait_thread(process, event_timer):
        process.wait()
        event_timer.set()

    def enqueue_output(out, log_queue):
        for line in iter(out.readline, b""):
            log_queue.put(line.decode("utf-8"))
        out.close()
        return

    def start_thread(fn, *args):
        stdout_t = threading.Thread(target=fn, args=args)
        stdout_t.daemon = True
        stdout_t.start()

    def print_subprocess_log(log_queue):
        while not log_queue.empty():
            print((log_queue.get()).strip())

    def run_cmd_with_timeout(cmd):
        os.chdir(str(TEST_DIR))
        stdout_queue = queue.Queue()
        event_timer = threading.Event()

        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
        start_thread(wait_thread, p, event_timer)

        try:
            event_timer.wait(2000)
            ret = p.poll()
            if ret:
                print_subprocess_log(stdout_queue)
            if not event_timer.is_set():
                ret = 1
                p.kill()
                p.terminate()
                print("Timeout: Command '{}' timed out after 2000 seconds".format(" ".join(cmd)))
                print_subprocess_log(stdout_queue)
        except Exception as err:
            ret = 1
            print(err)
        return ret

    def run_tests(files):
        exec_infos = []
        has_failed = 0
        for ut_type, ut_files in files.items():
            for ut_file in ut_files:
                if not os.path.basename(ut_file).startswith("test_"):
                    continue
                cmd = get_ut_cmd(ut_type, ut_file)
                ut_info = " ".join(cmd[4:]).replace(" -- -k", "")
                ret = run_cmd_with_timeout(cmd)
                if ret:
                    has_failed = ret
                    exec_infos.append("exec ut {} failed.".format(ut_info))
                else:
                    exec_infos.append("exec ut {} success.".format(ut_info))
        return has_failed, exec_infos

    ret_status, exec_infos = run_tests(files)

    print("***** Total result:")
    for exec_info in exec_infos:
        print(exec_info)
    return ret_status


if __name__ == "__main__":
    cur_modify_files = str(os.path.join(BASE_DIR, "modify_files.txt"))
    test_mgr = TestMgr()
    test_mgr.load(cur_modify_files)
    test_mgr.analyze()
    cur_test_files = test_mgr.get_test_files()

    test_mgr.print_modify_files()
    test_mgr.print_ut_files()
    test_mgr.print_op_ut_files()

    ret_ut = exec_ut(cur_test_files)
    sys.exit(ret_ut)
