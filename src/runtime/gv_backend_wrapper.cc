#include "gptvm/runtime/backend_wrapper.h"

#include <Python.h>
#include <cassert>
#include <dirent.h>
#include <iostream>
#include <numeric>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <unordered_map>

namespace gptvm {

bool GVBackendWrapper::loadBackendLib(std::string backend_path,
                                      GVDevice device) {
  dlhandler = dlopen(backend_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (dlhandler == NULL)
    return false;

  void *ptr = dlsym(dlhandler, "add_backend");
  if (ptr == NULL)
    return false;

  auto func = reinterpret_cast<addBackend>(ptr);
  backend = func();
  if (backend == nullptr)
    return false;

  // check device whether supported by backend
  auto device_list = backend->getDeviceList();
  // assert(device_list.size());

  if (std::find(device_list.begin(), device_list.end(), device) !=
      device_list.end()) {
    for (auto &device : device_list)
      backend->register_allocator(device.device_type);

    return true;
  }

  delete backend;
  backend = nullptr;
  dlclose(dlhandler);
  dlhandler = NULL;
  return false;
}

void GVBackendWrapper::getBackend(std::string backend_name, GVDevice device) {

  if (dlhandler)
    return;

  if (!backend_name.empty()) {
    assert(
        loadBackendLib(backend_path + "/lib" + backend_name + ".so", device));
    return;
  }

  struct stat backend_stat;
  assert(stat(backend_path.c_str(), &backend_stat) != -1);
  assert(backend_stat.st_mode & S_IFDIR);

  DIR *backend_dir = opendir(backend_path.c_str());
  assert(backend_dir);
  struct dirent *dirp;

  while (((dirp = readdir(backend_dir)) != NULL)) {
    struct stat fs;
    auto lib_name = backend_path + "/" + std::string(dirp->d_name);
    if (stat(lib_name.c_str(), &fs) == -1)
      continue;
    if (fs.st_mode & S_IFREG == 0)
      continue;
    if (fs.st_mode & S_IFDIR)
      continue;

    std::string backend_name = dirp->d_name;
    if (backend_name.at(0) == 'l' && backend_name.at(1) == 'i' &&
        backend_name.at(2) == 'b')
      backend_name.erase(backend_name.begin(),
                         backend_name.begin() + std::string("lib").length());

    if (backend_name.at(backend_name.length() - 3) == '.' &&
        backend_name.at(backend_name.length() - 2) == 's' &&
        backend_name.at(backend_name.length() - 1) == 'o')
      backend_name.erase(backend_name.end() - std::string(".so").length(),
                         backend_name.end());

    if (loadBackendLib(lib_name, device)) {
      this->backend_name = backend_name;
      return;
    }
  }
  assert(false);
}

static void getInputShape(
    std::unordered_map<std::string, std::pair<int64_t, size_t>> inputs,
    GVNamedShape &inputs_shape, GVNamedTensorInfo model_inputs) {
  for (auto &[name, data] : inputs) {
    assert(model_inputs.count(name) > 0);

    auto &model_input = model_inputs[name];
    // If user does not set shape of input, get shape from model.
    if (inputs_shape[name].size() == 0)
      inputs_shape[name] = model_input.shape;
    auto input_shape = inputs_shape[name];
    assert(input_shape.size() == model_input.shape.size());

    // Index of dynamic dim. If only one dynamic dim in shape, the dim can be
    // calculate by data size and other dims.
    int dynamic_dim = -1;
    size_t shape_size = 1;
    for (int i = 0; i < input_shape.size(); i++) {
      if (input_shape[i] == -1) {
        if (dynamic_dim != -1)
          assert(false);
        dynamic_dim = i;
      } else
        shape_size *= input_shape[i];

      if (model_input.shape[i] == -1)
        continue;
      assert(input_shape[i] == model_input.shape[i]);
    }

    if (dynamic_dim != -1) {
      auto input_size = data.second;

      assert(input_size % (model_input.elem_size * shape_size) == 0);
      input_shape[dynamic_dim] =
          input_size / (model_input.elem_size * shape_size);
      inputs_shape[name] = input_shape;
    }
  }
}

GVNamedOutput GVBackendWrapper::forward(
    std::unordered_map<std::string, std::pair<int64_t, size_t>> args,
    GVNamedShape inputs_shape, bool use_cache) {

  char cCurrentPath[FILENAME_MAX];
  static GVModel model_info_built;

  getcwd(cCurrentPath, sizeof(cCurrentPath));

  getInputShape(args, inputs_shape, model_info.inputs_info);

  // if the model has dynamic shape and use_cache is true, we take this as a
  // hint to build the model first
  if (use_cache == false) {
    getBackend(backend_name, device);
    model_info_built = backend->build(model_data, model_size, inputs_shape,
                                      device.device_type);
    model_info.handler = model_info_built.handler;
  }

  GVNamedBuffer inputs;

  for (auto &[name, input_data] : args) {
    auto size = input_data.second;
    auto src_data = (char *)input_data.first;
    char *src_data_resize = NULL;

    // Convert int64 to int32
    if (model_info.inputs_info[name].elem_size !=
        model_info_built.inputs_info[name].elem_size) {
      // TODO: Require to validate type of element.
      assert(model_info.inputs_info[name].elem_size == 8 &&
             model_info_built.inputs_info[name].elem_size == 4);

      auto resize_size = size * sizeof(int64_t) / sizeof(int32_t);
      src_data_resize = new char[resize_size];
      for (size_t i = 0; i < size / sizeof(int64_t); i++) {
        auto value = ((int64_t *)src_data)[i];
        if (value > static_cast<int64_t>(INT32_MAX) ||
            value < static_cast<int64_t>(INT32_MIN))
          assert(false);
        else
          ((int32_t *)src_data_resize)[i] = static_cast<int32_t>(value);
      }

      src_data = src_data_resize;
      size = resize_size;
    }

    auto dev_data = backend->memAlloc(size, device);
    assert(dev_data);
    auto buffer = GVBuffer(dev_data, size, device);
    auto ret =
        backend->memCopy(dev_data, src_data, size, GV_MEMCPY_H2D, device);
    assert(ret == size);
    inputs[name] = buffer;

    if (src_data_resize)
      delete src_data_resize;
  }

  // Validate output built info.
  auto &outputs_model_info = model_info.outputs_info;
  auto &outputs_built_info = model_info_built.outputs_info;
  if (outputs_built_info.size())
    assert(outputs_model_info.size() == outputs_built_info.size());
  else
    outputs_built_info = outputs_model_info;

  for (auto &[name, built_info] : outputs_built_info) {
    assert(outputs_model_info.count(name) > 0);
    auto model_info = outputs_model_info[name];
    for (int i = 0; i < built_info.shape.size(); i++) {
      // assert(built_info.shape[i] > 0);
      if (model_info.shape[i] > 0)
        assert(built_info.shape[i] == model_info.shape[i]);
    }
  }

  auto outputs = backend->run(model_info_built, inputs_shape, inputs, device);

  // Free inputs and outputs.
  for (auto &[name, buffer] : inputs) {
    backend->memFree(buffer.impl->data, device);
    buffer.impl->size = 0;
  }

  for (auto &[name, buffer] : outputs) {
    if (model_info.outputs_info[name].elem_size !=
        model_info_built.outputs_info[name].elem_size)
      assert(false);
  }

  return outputs;
}

void GVBackendWrapper::memFree(
    std::vector<std::pair<int64_t, uint32_t>> mems_info) {
  for (auto &mem_info : mems_info) {
    backend->memFree((char *)mem_info.first, GVDevice(mem_info.second));
  }
}

} // namespace gptvm
