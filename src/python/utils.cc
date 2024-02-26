#include "gptvm/python/utils.h"

#define Py_CPYTHON_FRAMEOBJECT_H
#include <cpython/frameobject.h>

#include <unordered_map>

namespace gptvm {

std::tuple<PyObject *, PyObject *, PyObject *>
extractArgs(PyFrameObject *frame) {
  PyCodeObject *code = PyFrame_GetCode(frame);
  int argcount = code->co_argcount + code->co_kwonlyargcount;

  PyObject *varargs = nullptr;
  int varargcount = 0;
  if (code->co_flags & CO_VARARGS) {
    varargs = frame->f_localsplus[argcount];
    if (PyTuple_Check(varargs)) {
      varargcount = PyTuple_Size(varargs);
    }
  }
  assert(varargcount >= 0);

  std::unordered_map<int, int> arg2cell;
  for (int i = 0; i < PyTuple_Size(code->co_cellvars); i++) {
    if (auto *cell2arg = code->co_cell2arg) {
      if (cell2arg[i] != -1) {
        arg2cell[code->co_cell2arg[i]] = i;
      }
    }
  }

  PyObject **cellvars = frame->f_localsplus + code->co_nlocals;
  PyObject *args = PyTuple_New(argcount + varargcount);
  for (int i = 0; i < argcount; i++) {
    PyObject *arg = frame->f_localsplus[i];
    if (arg == nullptr) {
      PyObject *cell = cellvars[arg2cell.at(i)];
      arg = PyCell_GET(cell);
    }
    Py_INCREF(arg);
    PyTuple_SetItem(args, i, arg);
  }
  for (int i = 0; i < varargcount; i++) {
    PyObject *arg = PyTuple_GetItem(varargs, i);
    if (arg == nullptr) {
      PyObject *cell = cellvars[arg2cell.at(argcount + i)];
      arg = PyCell_GET(cell);
    }
    Py_INCREF(arg);
    PyTuple_SetItem(args, argcount + i, arg);
  }

  PyObject *kwargs = nullptr;
  if (code->co_flags & CO_VARKEYWORDS) {
    int kwarg_idx = argcount + (varargs != nullptr);
    auto it = arg2cell.find(kwarg_idx);
    if (it != arg2cell.end()) {
      PyObject *cell = cellvars[it->second];
      kwargs = PyCell_GET(cell);
    } else {
      kwargs = frame->f_localsplus[kwarg_idx];
    }
  }

  PyObject *freevars = PyTuple_New(PyTuple_Size(code->co_freevars));
  int ncellvars = PyTuple_Size(code->co_cellvars);
  for (int i = 0; i < PyTuple_Size(code->co_freevars); i++) {
    PyObject *o = *(cellvars + ncellvars + i);
    Py_INCREF(o);
    PyTuple_SetItem(freevars, i, o);
  }

  return std::make_tuple(args, kwargs, freevars);
}

PyObject *importModuleObject(const char *module, const char *object) {
  PyObject *module_name = PyUnicode_FromString(module);
  PyObject *module_obj = PyImport_Import(module_name);
  if (!module_obj) {
    PyErr_Print();
    assert(false);
  }
  return PyObject_GetAttrString(module_obj, object);
}

PyObject *getClassType(PyFrameObject *frame) {
  PyCodeObject *code = PyFrame_GetCode(frame);

  if (code->co_argcount < 1) {
    return nullptr;
  }
  if (code->co_nlocals < 1 || PyTuple_Size(code->co_varnames) < 1)
    return nullptr;

  PyObject *first_var = frame->f_localsplus[0];
  if (!first_var)
    return nullptr;

  PyObject *first_var_name = PyTuple_GetItem(code->co_varnames, 0);
  if (strcmp(PyUnicode_AsUTF8(first_var_name), "self") == 0) {
    return reinterpret_cast<PyObject *>(first_var->ob_type);
  } else if (strcmp(PyUnicode_AsUTF8(first_var_name), "cls") == 0) {
    if (PyType_Check(first_var))
      return first_var;
  }
  return nullptr;
}

} // namespace gptvm
