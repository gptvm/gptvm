#pragma once

#include <Python.h>

#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

namespace gptvm {

class TorchModel {
public:
  explicit TorchModel(PyObject *cls) : cls(cls) {}

  PyObject *forward(PyObject *args, PyObject *kwargs);

  class Param {
  public:
    virtual std::unique_ptr<Param> update(PyObject *arg) = 0;

    virtual bool match(PyObject *arg) = 0;

    virtual void set_dynamic_axes(PyObject *arg) = 0;

    virtual void dump() = 0;

    static std::unique_ptr<Param> create(PyObject *arg);
  };

  class GeneralParam : public Param {
  public:
    GeneralParam(PyObject *arg) : value(arg) {}

    std::unique_ptr<Param> update(PyObject *arg);

    bool match(PyObject *arg);

    void set_dynamic_axes(PyObject *arg) {}

    void dump() {
      std::cerr << "<GeneralParam>(";
      PyObject_Print(value, stderr, 0);
      std::cerr << ")";
    }

  private:
    PyObject *value;
  };

  class TensorParam : public Param {
  public:
    TensorParam(PyObject *arg) { shape = getTorchTensorShape(arg); }

    TensorParam(const std::vector<int64_t> shape) : shape(shape) {}

    std::unique_ptr<Param> update(PyObject *arg);

    bool match(PyObject *arg);

    void set_dynamic_axes(PyObject *arg);

    void dump() {
      std::cerr << "<TensorParam>(";
      for (size_t i = 0; i < shape.size(); i++) {
        if (i != 0)
          std::cerr << ", ";
        std::cerr << shape[i];
      }
      std::cerr << ")";
    }

  private:
    std::vector<int64_t> shape;
  };

  class ListParam : public Param {
  public:
    ListParam(PyObject *arg) {
      for (size_t i = 0; i < PyTuple_Size(arg); i++)
        params.emplace_back(Param::create(PyTuple_GetItem(arg, i)));
    }

    std::unique_ptr<Param> update(PyObject *arg);

    bool match(PyObject *arg);

    void set_dynamic_axes(PyObject *arg);

    void dump() {
      std::cerr << "<ListParam>(";
      for (size_t i = 0; i < params.size(); i++) {
        if (i != 0)
          std::cerr << ", ";
        params[i]->dump();
      }
      std::cerr << ")";
    }

  private:
    std::vector<std::unique_ptr<Param>> params;
  };

private:
  static std::vector<int64_t> getTorchTensorShape(PyObject *tensor) {
    std::vector<int64_t> shape;
    PyObject *size = PyObject_CallMethod(tensor, "size", nullptr);

    for (size_t i = 0; i < PyTuple_Size(size); i++) {
      shape.push_back(PyLong_AsLong(PyTuple_GetItem(size, i)));
    }
    return shape;
  }

  PyObject *cls;

  std::unique_ptr<ListParam> model_params;

  int64_t forward_count = 0;

  PyObject *optimized_forward = nullptr;
};

class TorchModelOptimizer {
public:
  PyObject *optimize(PyFrameObject *frame);

private:
  std::unordered_map<PyObject *, TorchModel> models;
};

} // namespace gptvm
