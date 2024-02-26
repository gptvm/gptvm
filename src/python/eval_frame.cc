#include "gptvm/python/gptvm.h"
#include "gptvm/python/torch_opt.h"
#include "gptvm/python/utils.h"

namespace gptvm {

static inline bool fastStrCheck(PyObject *obj, const char *str) {
  const char *obj_str = reinterpret_cast<char *>(PyUnicode_1BYTE_DATA(obj));
  if (strlen(obj_str) != strlen(str))
    return false;
  return strcmp(obj_str, str) == 0;
}

PyObject *FrameEvaluator::eval(PyFrameObject *frame) {
  PyCodeObject *code = PyFrame_GetCode(frame);
  auto *extra = reinterpret_cast<PyObject *>(code->co_extra);
  if (extra && PyFunction_Check(extra)) {
    auto [args, kwargs, cellvars] = extractArgs(frame);
    auto *result = PyObject_Call(extra, args, kwargs);
    Py_DECREF(args);
    Py_DECREF(cellvars);
    return result;
  }

  if (GVConfig::get().capture_torch_forward &&
      fastStrCheck(code->co_name, "forward")) {
    return torch_optimizer.optimize(frame);
  }

  return nullptr;
}

} // namespace gptvm
