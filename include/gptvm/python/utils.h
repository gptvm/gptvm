#pragma once

#include <Python.h>

#include <tuple>

namespace gptvm {

std::tuple<PyObject *, PyObject *, PyObject *>
extractArgs(PyFrameObject *frame);

PyObject *importModuleObject(const char *module, const char *object);

PyObject *getClassType(PyFrameObject *frame);

} // namespace gptvm
