#pragma once

#include "gptvm/runtime/device.h"

#include <cassert>
#include <memory>
#include <string>
#include <vector>

namespace gptvm {

/// Type to copy memory.
enum GVMemcpyType : uint8_t {
  /// Type to copy memory from host to device.
  GV_MEMCPY_H2D,
  /// Type to copy memory from device to host.
  GV_MEMCPY_D2H,
  /// Type to copy memory from device to device.
  GV_MEMCPY_D2D,
};

/// GVDeviceAllocator is the base class for all kinds of backends to provide
/// memory functions.
class GVDeviceAllocator {
public:
  /// Allocate memory in the deivce.
  /// @param size Byte size of allocated memory.
  /// @param device Device in which to allocate memory.
  /// @return Address of allocated memory, nullptr for failure.
  virtual char *memAlloc(size_t size, GVDevice device) = 0;

  /// Free memory in the device.
  /// @param mem Address of memory to free.
  /// @param device Device in which to free memeory.
  virtual void memFree(char *mem, GVDevice device) = 0;

  /// Copy memory between devices or device and host.
  /// TODO: require one more device argument for D2D type.
  /// \param dest Destination memory address.
  /// \param src Source memory address.
  /// \param size Byte size to copy memory.
  /// \param type Type to copy memory.
  /// \param device Device to copy memory.
  /// \return Size of copied memory.
  virtual size_t memCopy(char *dest, char *src, size_t size, GVMemcpyType type,
                         GVDevice device) = 0;
};

}; // namespace gptvm
