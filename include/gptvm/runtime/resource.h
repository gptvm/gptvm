#pragma once

#include "gptvm/runtime/device.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace gptvm {

/// define the singletion calss for resource of device reside in the node
class GVResource {
public:
  virtual ~GVResource() = default;
  // forbid copy and assign
  // GVResource(const GVResource &) = delete;
  GVResource &operator=(const GVResource &) = delete;

  /// get the resources of certain type
  // GVResource& getResource(GVDeviceType type) const;

  static GVResource &getInstance() { return instance; }

  /// get the devices
  const std::vector<GVDevice> &getDevices(GVDeviceType dev_type) const;
  /// get the compute capability
  uint64_t getComputeCapability(GVDevice dev, std::string data_type) const;
  const std::unordered_map<std::string, float>
  getComputeCapability(GVDevice dev) const;
  /// get the memory size
  uint64_t getMemorySize(GVDevice dev) const;

private:
  uint16_t node_id;
  std::unordered_map<GVDeviceType, std::vector<GVDevice>> devices_map;
  std::unordered_map<GVDevice, std::unordered_map<std::string, float>,
                     GVDeviceHash>
      compute_cap;
  // for each device, the memory size
  // NOTE, for the host device, we only have the device type and node id
  // for the map key, the device_id is ignored
  // for GPU, both the device_id and device_type are used
  std::unordered_map<GVDevice, uint64_t, GVDeviceHash> memory_size_map; // bytes
  // TODO, device mesh info

  static GVResource instance;
  GVResource() { resourceInit(); };

  // the real work to init the resource
  void resourceInit();
};

} // namespace gptvm
