#pragma once

#include "gptvm/backend/cpu.h"
#include "gptvm/backend/device.h"
#include "gptvm/backend/gpu.h"
#include "gptvm/runtime/buffer.h"
#include "gptvm/runtime/model.h"

#include <dlfcn.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace gptvm {

using GVNamedBuffer = std::unordered_map<std::string, GVBuffer>;
using GVNamedShape = std::unordered_map<std::string, std::vector<int64_t>>;
using GVNamedElemSize = std::unordered_map<std::string, int>;
using GVNamedOutput =
    std::unordered_map<std::string,
                       std::tuple<int64_t, size_t, std::vector<int64_t>>>;

/// GVBackend provides inference functions.
class GVBackend : public GVDeviceAllocator {
public:
  // Return handle of dlopen used to load backend dynamically.
  void *dlhandler = nullptr;
  ~GVBackend() {
    for (auto &[_, allocator] : allocators)
      delete allocator;

    if (dlhandler)
      dlclose(dlhandler);
  }

  /// Get available device list in the backend.
  /// \return List of available devices
  virtual std::vector<GVDevice> getDeviceList(void) = 0;

  /// Build model in the backend.
  /// \param model_data Data of model file.
  /// \param input_shapes List of inputs's shapes.
  /// \param device_type Type of device.
  /// \return Information of model.
  virtual GVModel build(char *model_data, size_t model_size,
                        const GVNamedShape &input_shapes,
                        GVDeviceType device_type) = 0;

  /// Run model in the backend.
  /// \param model_info Information of model file.
  /// \param input_shapes List of inputs's shapes.
  /// \param inputs List of model inputs.
  /// \param device Device ID.
  /// \return Buffer information of outputs.
  virtual GVNamedOutput run(GVModel model_info,
                            const GVNamedShape &input_shapes,
                            const GVNamedBuffer &inputs, GVDevice device) = 0;

  /// Allocate memory in the deivce.
  /// @param size Byte size of allocated memory.
  /// @param device Device in which to allocate memory.
  /// @return Address of allocated memory, nullptr for failure.
  virtual char *memAlloc(size_t size, GVDevice device) override {
    auto allocator = allocators[device.device_type];
    return allocator->memAlloc(size, device);
  }

  /// Free memory in the device.
  /// @param mem Address of memory to free.
  /// @param device Device in which to free memeory.
  virtual void memFree(char *mem, GVDevice device) override {
    auto allocator = allocators[device.device_type];
    allocator->memFree(mem, device);
  }

  /// Copy memory between devices or device and host.
  /// TODO: require one more device argument for D2D type.
  /// \param dest Destination memory address.
  /// \param src Source memory address.
  /// \param size Byte size to copy memory.
  /// \param type Type to copy memory.
  /// \param device Device to copy memory.
  /// \return Size of copied memory.
  virtual size_t memCopy(char *dest, char *src, size_t size, GVMemcpyType type,
                         GVDevice device) override {
    auto allocator = allocators[device.device_type];
    return allocator->memCopy(dest, src, size, type, device);
  }

  /// Register an allocator used by backend.
  /// \param type Type of device.
  /// \param allocator Pointer to allocator object.
  void register_allocator(GVDeviceType type,
                          GVDeviceAllocator *allocator = nullptr) {
    if (allocators.count(type))
      return;

    if (allocator != nullptr) {
      allocators[type] = allocator;
      return;
    }

    switch (type) {
    case GV_CPU:
      allocator = new GVCPUAllocator();
      assert(allocator);
      break;

    case GV_NV_GPU:
      allocator = new GVGPUAllocator();
      break;

    default:
      assert(false);
    }

    assert(allocator);
    allocators[type] = allocator;
  }

private:
  std::unordered_map<GVDeviceType, GVDeviceAllocator *> allocators;
};

/// Each backend should provide c style functions to register and unregister to
/// backend core.
extern "C" {
/// Create backend instance, this function is implemented mandatorily in the
/// backend. Function defination is: std::shared_ptr<gptvm::GVBackend>
/// *add_backend(void)
typedef GVBackend *(*addBackend)(void);
}

}; // namespace gptvm
