## Description
The `csrc` lib implements python interface, which use `pybind11` to wrap the C++ code.
There are 3 files you need to focus:
1. `pybind.cpp`: Define the python interface.
2. `functions.cpp`: Define the C++ interface.
3. The file naming in `Pascal` style: The implementation of the C++ interface.