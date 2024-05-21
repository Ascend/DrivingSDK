import argparse
import pathlib
import os
import sys
import signal
import math
from datetime import datetime, timezone
from typing import Optional, List

import torch
from torch.testing._internal.common_utils import shell

import torch_npu

TEST_ROOT = pathlib.Path(__file__).resolve().parent

# https://stackoverflow.com/questions/2549939/get-signal-names-from-numbers-in-python
SIGNALS_TO_NAMES_DICT = {
    getattr(signal, n): n for n in dir(signal) if
    n.startswith("SIG") and "_" not in n
}


def print_to_stderr(message):
    print(message, file=sys.stderr)


def discover_tests(
        base_dir: Optional[pathlib.Path] = None,
        blocklisted_patterns: Optional[List[str]] = None,
        blocklisted_tests: Optional[List[str]] = None,
        extra_tests: Optional[List[str]] = None) -> List[str]:
    """
    Searches for all python files starting with test_ excluding one specified by patterns
    """

    def skip_test_p(name: str) -> bool:
        rc = False
        if blocklisted_patterns is not None:
            rc |= any(
                name.startswith(pattern) for pattern in blocklisted_patterns)
        if blocklisted_tests is not None:
            rc |= name in blocklisted_tests
        return rc

    cwd = pathlib.Path(
        __file__).resolve().parent if base_dir is None else base_dir
    all_py_files = list(cwd.glob('**/test_*.py'))
    rc = [str(fname.relative_to(cwd))[:-3] for fname in all_py_files]
    rc = [test for test in rc if not skip_test_p(test)]
    if extra_tests is not None:
        rc += extra_tests
    return sorted(rc)


def parse_test_module(test):
    return pathlib.Path(test).parts[0]


TESTS = discover_tests(
    blocklisted_patterns=[],
    blocklisted_tests=[],
    extra_tests=[]
)

TESTS_MODULE = list(set([parse_test_module(test) for test in TESTS]))

TEST_CHOICES = TESTS + TESTS_MODULE


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the PyTorch unit test suite",
        epilog="where TESTS is any of: {}".format(", ".join(TESTS)),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="print verbose information and test-by-test results",
    )
    parser.add_argument(
        "-i",
        "--include",
        nargs="+",
        choices=TEST_CHOICES,
        default=TESTS,
        metavar="TESTS",
        help="select a set of tests to include (defaults to ALL tests)."
             " tests must be a part of the TESTS list defined in run_test.py",
    )
    parser.add_argument(
        "additional_unittest_args",
        nargs="*",
        help="additional arguments passed through to unittest, e.g., "
             "python run_test.py -i sparse -- TestSparse.test_factory_size_check",
    )
    return parser.parse_args()


def get_selected_tests(options):
    selected_tests = []
    if options.include:
        for item in options.include:
            selected_tests.extend(
                list(filter(lambda test_name: item == test_name \
                                              or (
                                                          item in TESTS_MODULE and test_name.startswith(
                                                      item)), TESTS)))
    else:
        selected_tests = TESTS
    return selected_tests


def run_test(test, test_directory, options):
    unittest_args = options.additional_unittest_args.copy()

    if options.verbose:
        unittest_args.append("-v")
    # get python cmd.
    executable = [sys.executable]

    # Can't call `python -m unittest test_*` here because it doesn't run code
    # in `if __name__ == '__main__': `. So call `python test_*.py` instead.
    argv = [test + ".py"] + unittest_args

    command = executable + argv
    print_to_stderr(
        "Executing {} ... [{}]".format(command, datetime.now(tz=timezone.utc)))
    return shell(command, test_directory)


def run_test_module(test: str, test_directory: str, options) -> Optional[str]:
    print_to_stderr(
        "Running {} ... [{}]".format(test, datetime.now(tz=timezone.utc)))

    return_code = run_test(test, test_directory, options)
    if not (isinstance(return_code, int) and not isinstance(return_code, bool)):
        raise TypeError("Return code should be an integer")
    if return_code == 0:
        return None

    message = f"exec ut {test} failed!"
    if return_code < 0:
        # subprocess.Popen returns the child process' exit signal as
        # return code -N, where N is the signal number.
        signal_name = SIGNALS_TO_NAMES_DICT[-return_code]
        message += f" Received signal: {signal_name}"
    return message


def main():
    options = parse_args()
    selected_tests = get_selected_tests(options)

    if options.verbose:
        print_to_stderr("Selected tests: {}".format(", ".join(selected_tests)))

    has_failed = False
    failure_msgs = []

    for test in selected_tests:
        err_msg = run_test_module(test, TEST_ROOT, options)

        if err_msg is None:
            continue
        has_failed = True
        failure_msgs.append(err_msg)
        print_to_stderr(err_msg)

    if has_failed:
        for err in failure_msgs:
            print_to_stderr(err)
        return False
    return True


if __name__ == "__main__":
    if not main():
        sys.exit(1)
