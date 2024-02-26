#include "gptvm/python/gptvm.h"
#include "gptvm/runtime/backend_wrapper.h"

#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include <Python.h>

namespace py = pybind11;

namespace gptvm {

/// Export gptvm python module
PYBIND11_MODULE(libgptvm, m) {
  /// Export GVBackendWrapper class.
  py::class_<GVBackendWrapper>(m, "GVBackend")
      .def(py::init<uint32_t, std::string, std::string,
                    std::pair<int64_t, size_t>, GVNamedShape, GVNamedElemSize,
                    GVNamedShape, GVNamedElemSize>(),
           py::arg("device_id"), py::arg("backend_name"),
           py::arg("backend_path"), py::arg("model_data"),
           py::arg("model_inputs_shape"), py::arg("model_inputs_elem_size"),
           py::arg("model_outputs_shape"), py::arg("model_outputs_elem_size"))
      .def("forward", &GVBackendWrapper::forward, py::arg("args"),
           py::arg("inputs_shape"), py::arg("use_cache"))
      .def("memFree", &GVBackendWrapper::memFree, py::arg("mems_info"));

  py::class_<GVConfig>(m, "GVConfig")
      .def_readwrite("debug", &GVConfig::debug)
      .def_readwrite("enable_injector", &GVConfig::enable_injector)
      .def_readwrite("capture_torch_forward", &GVConfig::capture_torch_forward);

  m.def(
      "get_config", []() -> GVConfig & { return GVConfig::get(); },
      py::return_value_policy::reference);

  /// Export frame call injection registeration function
  m.def("register_injector", [](py::object code, py::function func) {
    Py_INCREF(func.ptr());
    auto *code_obj = reinterpret_cast<PyCodeObject *>(code.ptr());
    code_obj->co_extra = func.ptr();
  });

  /// Attach custom frame evaluation function
  auto eval_frame = [](PyThreadState *tstate, PyFrameObject *frame,
                       int throwflag) {
    auto &config = GVConfig::get();
    if (config.enable_injector) {
      config.enable_injector = false;
      auto *rval = FrameEvaluator::get().eval(frame);
      config.enable_injector = true;
      if (rval)
        return rval;
    }
    return _PyEval_EvalFrameDefault(tstate, frame, throwflag);
  };

  auto ts = PyThreadState_Get();
  _PyInterpreterState_SetEvalFrameFunc(ts->interp, eval_frame);
}

} // namespace gptvm
