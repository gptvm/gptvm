#pragma once

#include "gptvm/backend/device.h"

#include <cuda_runtime_api.h>

namespace gptvm {
/// \brief this is a class to provide common functions related to cuda
class GVGPUAllocator : public GVDeviceAllocator {
public:
  /// \brief allocate memory, call cudaMalloc
  /// \param size: the size of the memory
  /// \param device: the device to allocate the memory, GPU only
  char *memAlloc(size_t size, GVDevice device) override {
    char *ptr = nullptr;
    cudaSetDevice(device.device_id);
    cudaMalloc((void **)&ptr, size);
    return ptr;
  }

  /// \brief free memory, call cudaFree
  /// \param ptr: the pointer to the memory
  /// \param device: the device to free the memory, GPU only
  void memFree(char *ptr, GVDevice device) override {
    cudaSetDevice(device.device_id);
    cudaFree(ptr);
  }

  /// \brief copy memory, call cudaMemcpy
  /// \param dst: the destination pointer
  /// \param src: the source pointer
  /// \param size: the size of the memory
  /// \param type: the type of the copy
  /// \param device: the device to copy the memory, GPU only
  size_t memCopy(char *dst, char *src, size_t size, GVMemcpyType type,
                 GVDevice device) override {
    cudaMemcpyKind memcpyType;
    switch (type) {
    case GV_MEMCPY_H2D:
      memcpyType = cudaMemcpyHostToDevice;
      break;
    case GV_MEMCPY_D2H:
      memcpyType = cudaMemcpyDeviceToHost;
      break;
    case GV_MEMCPY_D2D:
      memcpyType = cudaMemcpyDeviceToDevice;
      break;
    default:
      break;
    }
    cudaSetDevice(device.device_id);
    cudaMemcpy(dst, src, size, memcpyType);
    return size;
  }
};

} // namespace gptvm
