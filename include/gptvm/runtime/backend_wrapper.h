#pragma once

#include "gptvm/backend/backend.h"
#include "gptvm/runtime/device.h"
#include "gptvm/runtime/model.h"

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;
namespace gptvm {

/// GVBackendWrapper provides interfaces for user to run on backend.
class GVBackendWrapper {
public:
  GVBackendWrapper(uint32_t device_id, std::string backend_name,
                   std::string backend_path,
                   std::pair<int64_t, size_t> model_data,
                   GVNamedShape model_inputs_shape,
                   GVNamedElemSize model_inputs_elem_size,
                   GVNamedShape model_outputs_shape,
                   GVNamedElemSize model_outputs_elem_size)
      : device(GVDevice(device_id)), backend_name(backend_name),
        backend_path(backend_path), model_data((char *)model_data.first),
        model_size(model_data.second) {

    for (auto [name, shape] : model_inputs_shape)
      model_info.inputs_info[name] =
          GVTensorInfo(model_inputs_elem_size[name], shape);

    for (auto [name, shape] : model_outputs_shape)
      model_info.outputs_info[name] =
          GVTensorInfo(model_outputs_elem_size[name], shape);

    dlhandler = NULL;
    backend = nullptr;
  }

  ~GVBackendWrapper() {
    delete model_data;

    if (dlhandler) {
      if (backend)
        delete backend;
      dlclose(dlhandler);
    }
  }

  /// Build and run an inference.
  /// \param args Input data of model.
  /// \param inputs_shape Shape of input data.
  /// \param use_cache Flag to reuse build cache.
  /// \return Output data.
  GVNamedOutput
  forward(std::unordered_map<std::string, std::pair<int64_t, size_t>> args,
          GVNamedShape inputs_shape, bool use_cache);

  /// Free memory
  /// \param mems_info memory address with its device
  void memFree(std::vector<std::pair<int64_t, uint32_t>> mems_info);

private:
  /// Load backend dynamic library.
  /// \param backend_path Path of backend dynamic library.
  /// @param device Device should be supported by backend.
  /// @return true to load a library, otherwize false.
  bool loadBackendLib(std::string backend_path, GVDevice device);

  /// Get backend object.
  /// \param backend_name Name of backend.
  /// \param device Device to run.
  void getBackend(std::string backend_name, GVDevice device);

  GVDevice device;
  std::string backend_name;
  std::string backend_path;
  char *model_data;
  size_t model_size;
  GVModel model_info;
  void *dlhandler;
  GVBackend *backend;
};

}; // namespace gptvm
