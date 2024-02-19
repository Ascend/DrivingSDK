#ifndef CSRC_PYBIND_H_
#define CSRC_PYBIND_H_
#include <pybind11/numpy.h>
void init_common(pybind11::module& m);
void init_motion(pybind11::module& m);
void init_percention_fused(pybind11::module& m);
void init_perception_point(pybind11::module& m);
void init_perception_vision(pybind11::module& m);
#endif  // CSRC_PYBIND_H_
