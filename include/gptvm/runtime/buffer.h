#pragma once

#include "gptvm/runtime/device.h"

#include <cassert>
#include <cstddef>
#include <memory>

namespace gptvm {

class GVBufferImpl {
public:
  GVBufferImpl(char *data, size_t size, GVDevice device)
      : data(data), size(size), device(device) {}

  ~GVBufferImpl() {
    // Do nothing for empty buffer.
    if (size == 0)
      return;

    // Free data in driver device.
    if (device == GVDevice(0)) {
      delete data;
      return;
    }
  }

  /// Pointer to the data.
  char *data;

  /// Data size in bytes.
  size_t size;

  /// Device where the buffer locates.
  GVDevice device;
};

/// GVBuffer reprensets a buffer of bytes on a certain device.
class GVBuffer {
public:
  std::shared_ptr<GVBufferImpl> impl;

  GVBuffer(char *data = NULL, size_t size = 0, GVDevice device = GVDevice(0))
      : impl(std::make_shared<GVBufferImpl>(data, size, device)) {}

  ~GVBuffer() { impl = nullptr; }
  /// Get data of the buffer.
  /// \return Pointer to the data.
  char *getData() const {
    assert(impl);
    return impl->data;
  }

  /// Get data size of the buffer.
  /// \return Data size in bytes.
  size_t getSize() const {
    assert(impl);
    return impl->size;
  }

  /// Get device where the buffer locates.
  /// \return Device where the buffer locates.
  GVDevice getDevice() const {
    assert(impl);
    return impl->device;
  }

  /// Set device where the buffer locates.
  /// \param device Device where the buffer locates.
  void setDevice(GVDevice device) {
    assert(impl);
    impl->device = device;
  }

  /// Allocate buffer in device and create an GVBuffer from allocated buffer.
  /// \param size Data size in bytes.
  /// \param device Device to locate buffer, default is driver device.
  /// \return Created GVBuffer.
  static GVBuffer create(size_t size, GVDevice device = GVDevice(0)) {
    if (size == 0)
      return GVBuffer(NULL, size, device);

    assert(device == GVDevice(0));
    auto data = new char[size];
    assert(data);
    return GVBuffer(data, size, device);
  }
};
} // namespace gptvm
