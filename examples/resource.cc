#include "gptvm/runtime/resource.h"

#include <iostream>

using namespace gptvm;

int main() {

  auto res = GVResource::getInstance();

  std::cout << "Resource info:" << std::endl;
  std::cout << "  CPU: " << std::endl;
  GVDevice cpu_device(0, GVDeviceType::GV_CPU, 0);
  for (auto &device : res.getDevices(GVDeviceType::GV_CPU)) {
    std::cout << "    " << gptvm::GVDeviceType2Name(device.device_type) << " "
              << device.node_id << ":" << (int)device.device_id << std::endl;
  }
  std::cout << "    Memory: " << res.getMemorySize(cpu_device) << std::endl;
  std::cout << "  GPU: " << std::endl;
  for (auto &device : res.getDevices(gptvm::GVDeviceType::GV_NV_GPU)) {
    std::cout << "    " << gptvm::GVDeviceType2Name(device.device_type) << " "
              << device.node_id << ":" << (int)device.device_id << std::endl;
    std::cout << "    Memory: " << res.getMemorySize(device) << std::endl;
    std::cout << "    Compute capability: " << std::endl;
    for (auto &cap : res.getComputeCapability(device)) {
      std::cout << "      " << cap.first << ": " << cap.second << std::endl;
    }
  }
}
