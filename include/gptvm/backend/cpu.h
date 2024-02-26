#pragma once

#include "gptvm/backend/device.h"

namespace gptvm {
/// \brief
class GVCPUAllocator : public GVDeviceAllocator {
public:
  /// \brief allocate memory, call cudaMalloc
  /// \param size: the size of the memory
  /// \param device: the device to allocate the memory, GPU only
  char *memAlloc(size_t size, GVDevice device) override {
    assert(size);
    auto ptr = new char[size];
    assert(ptr);
    return ptr;
  }

  /// \brief free memory, call cudaFree
  /// \param ptr: the pointer to the memory
  /// \param device: the device to free the memory, GPU only
  void memFree(char *ptr, GVDevice device) override {
    assert(ptr);
    delete ptr;
  }

  /// \brief copy memory, call cudaMemcpy
  /// \param dst: the destination pointer
  /// \param src: the source pointer
  /// \param size: the size of the memory
  /// \param type: the type of the copy
  /// \param device: the device to copy the memory, GPU only
  size_t memCopy(char *dst, char *src, size_t size, GVMemcpyType type,
                 GVDevice device) override {
    assert(src);
    assert(dst);
    std::copy(src, src + size, dst);
    return size;
  }
};

} // namespace gptvm
