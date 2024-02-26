#pragma once

#include <cstdint>
#include <map>
#include <string>

namespace gptvm {

/// Device type.
enum GVDeviceType : uint8_t {
  GV_CPU,
  GV_NV_GPU,
  GV_HW_ASCEND,
};

/// GVDevice reprensents a single device to execute tasks.
///
/// GVDevice uses a unique id to identify a device in the distributed
/// heterogenous compute system. GVDevice(0) is reserved as the driver
/// device, i.e. the CPU on the head node which starts the application.
union GVDevice {
  /// The unique global id in the system.
  uint32_t global_id;

  struct {
    /// Device id within a node.
    uint8_t device_id;

    /// Device type tag.
    GVDeviceType device_type;

    /// Node id in the distributed system.
    uint16_t node_id;
  };

  GVDevice(uint32_t global_id = 0) : global_id(global_id) {}

  GVDevice(uint16_t node_id, GVDeviceType device_type, uint8_t device_id)
      : node_id(node_id), device_type(device_type), device_id(device_id) {}

  bool operator==(const GVDevice &device) const {
    return global_id == device.global_id;
  }

  bool operator!=(const GVDevice &device) const {
    return global_id != device.global_id;
  }

  bool operator<(const GVDevice &device) const {
    return global_id < device.global_id;
  }
};

static std::string GVDeviceType2Name(GVDeviceType dtype) {
  switch (dtype) {
  case GVDeviceType::GV_CPU:
    return "CPU";
  case GVDeviceType::GV_NV_GPU:
    return "NV_GPU";
  case GVDeviceType::GV_HW_ASCEND:
    return "HW_ASCEND";
  default:
    return "UNKNOWN";
  }
};

struct GVDeviceHash {
  std::size_t operator()(const GVDevice &device) const {
    return device.global_id;
  }
};

} // namespace gptvm
