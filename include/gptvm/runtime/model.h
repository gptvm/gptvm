#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace gptvm {

/// GVTensorInfo provides information of model data.
struct GVTensorInfo {
  GVTensorInfo(int elem_size = 0,
               std::vector<int64_t> shape = std::vector<int64_t>())
      : elem_size(elem_size), shape(shape) {}

  /// Size of a data element.
  int elem_size;

  /// Shape of data.
  std::vector<int64_t> shape;
};

using GVNamedTensorInfo = std::unordered_map<std::string, GVTensorInfo>;

/// GVModel provides information of model used for model building and
/// running.
struct GVModel {

  /// handler of a model used to build and run.
  void *handler;

  /// list of named inputs' data information.
  GVNamedTensorInfo inputs_info;

  /// list of named outputs' data information.
  GVNamedTensorInfo outputs_info;
};

} // namespace gptvm
