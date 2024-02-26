#pragma once

#include "gptvm/python/torch_opt.h"

#include <Python.h>

#include <list>
#include <string>
#include <unordered_map>

namespace gptvm {

struct GVConfig {
  static GVConfig &get() {
    static GVConfig config;
    return config;
  }

  bool debug = false;

  bool enable_injector = true;

  bool capture_torch_forward = false;
};

class FrameEvaluator {
public:
  static FrameEvaluator &get() {
    static FrameEvaluator fe;
    return fe;
  }

  PyObject *eval(PyFrameObject *frame);

private:
  TorchModelOptimizer torch_optimizer;
};

} // namespace gptvm
