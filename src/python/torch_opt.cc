#include "gptvm/python/torch_opt.h"
#include "gptvm/python/utils.h"

namespace gptvm {

std::unique_ptr<TorchModel::Param> TorchModel::Param::create(PyObject *arg) {
  PyObject *torch_tensor = importModuleObject("torch", "Tensor");
  if (PyObject_IsInstance(arg, torch_tensor)) {
    return std::make_unique<TorchModel::TensorParam>(arg);
  } else if (PyTuple_Check(arg)) {
    return std::make_unique<TorchModel::ListParam>(arg);
  } else {
    return std::make_unique<TorchModel::GeneralParam>(arg);
  }
}

std::unique_ptr<TorchModel::Param>
TorchModel::GeneralParam::update(PyObject *arg) {
  PyObject *torch_tensor = importModuleObject("torch", "Tensor");
  if (PyObject_IsInstance(arg, torch_tensor)) {
    return std::make_unique<TorchModel::TensorParam>(arg);
  } else if (PyTuple_Check(arg)) {
    return std::make_unique<TorchModel::ListParam>(arg);
  } else {
    value = arg;
    return nullptr;
  }
}

bool TorchModel::GeneralParam::match(PyObject *arg) { return value == arg; }

std::unique_ptr<TorchModel::Param>
TorchModel::ListParam::update(PyObject *arg) {
  PyObject *torch_tensor = importModuleObject("torch", "Tensor");
  if (PyObject_IsInstance(arg, torch_tensor)) {
    return std::make_unique<TorchModel::TensorParam>(arg);
  } else if (PyTuple_Check(arg)) {
    if (params.size() != PyTuple_Size(arg))
      return std::make_unique<TorchModel::ListParam>(arg);
    for (size_t i = 0; i < PyTuple_Size(arg); i++) {
      if (auto &&newParam = params[i]->update(PyTuple_GetItem(arg, i))) {
        params[i] = std::move(newParam);
      }
    }
  }
  return nullptr;
}

bool TorchModel::ListParam::match(PyObject *arg) {
  if (!PyTuple_Check(arg))
    return false;
  if (params.size() != PyTuple_Size(arg))
    return false;
  for (size_t i = 0; i < PyTuple_Size(arg); i++) {
    if (!params[i]->match(PyTuple_GetItem(arg, i))) {
      return false;
    }
  }
  return true;
}

void TorchModel::ListParam::set_dynamic_axes(PyObject *arg) {
  for (size_t i = 0; i < PyTuple_Size(arg); i++) {
    params[i]->set_dynamic_axes(PyTuple_GetItem(arg, i));
  }
}

std::unique_ptr<TorchModel::Param>
TorchModel::TensorParam::update(PyObject *arg) {
  PyObject *torch_tensor = importModuleObject("torch", "Tensor");
  if (!PyObject_IsInstance(arg, torch_tensor)) {
    return nullptr;
  }

  auto newShape = getTorchTensorShape(arg);
  if (shape.size() != newShape.size())
    return std::make_unique<TorchModel::TensorParam>(newShape);

  for (size_t i = 0; i < shape.size(); i++) {
    if (shape[i] != newShape[i])
      shape[i] = -1;
  }
  return nullptr;
}

bool TorchModel::TensorParam::match(PyObject *arg) {
  PyObject *torch_tensor = importModuleObject("torch", "Tensor");
  if (!PyObject_IsInstance(arg, torch_tensor))
    return false;

  auto newShape = getTorchTensorShape(arg);
  if (shape.size() != newShape.size())
    return false;

  for (size_t i = 0; i < shape.size(); i++) {
    if (shape[i] != -1 && shape[i] != newShape[i])
      return false;
  }

  return true;
}

void TorchModel::TensorParam::set_dynamic_axes(PyObject *arg) {
  PyObject *mark_dynamic =
      importModuleObject("torch._dynamo", "maybe_mark_dynamic");
  PyObject *tuple = PyTuple_New(2);
  PyTuple_SetItem(tuple, 0, arg);
  for (size_t i = 0; i < shape.size(); i++) {
    if (shape[i] == -1) {
      PyTuple_SetItem(tuple, 1, PyLong_FromLong(i));
      PyObject_Call(mark_dynamic, tuple, nullptr);
    }
  }
}

PyObject *TorchModel::forward(PyObject *args, PyObject *kwargs) {
  if (kwargs)
    return nullptr;

  if (optimized_forward) {
    return PyObject_Call(optimized_forward, args, nullptr);
  }

  PyObject *compile =
      importModuleObject("gptvm.optimize.torch.compiler", "compile");
  PyObject *forward = PyObject_GetAttrString(cls, "forward");
  optimized_forward = PyObject_CallOneArg(compile, forward);
  if (auto *ret = PyObject_Call(optimized_forward, args, nullptr)) {
    return ret;
  }
  PyErr_Print();
  return nullptr;
}

PyObject *TorchModelOptimizer::optimize(PyFrameObject *frame) {
  PyObject *cls = getClassType(frame);
  if (!cls)
    return nullptr;

  PyObject *torch_module = importModuleObject("torch.nn", "Module");
  if (!PyObject_IsSubclass(cls, torch_module))
    return nullptr;

  auto [args, kwargs, cellvars] = extractArgs(frame);
  auto it = models.find(cls);
  if (it == models.end()) {
    it = models.insert({cls, TorchModel(cls)}).first;
  }
  if (PyObject *result = it->second.forward(args, kwargs)) {
    Py_DECREF(args);
    Py_DECREF(cellvars);
    return result;
  }

  auto *result =
      PyObject_Call(PyObject_GetAttrString(cls, "forward"), args, kwargs);
  Py_DECREF(args);
  Py_DECREF(cellvars);
  return result;
}

} // namespace gptvm
