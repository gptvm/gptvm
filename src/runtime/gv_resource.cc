#include "gptvm/runtime/resource.h"
#include "gptvm/runtime/runtime.h"

#include <cstring>
#if GV_HAS_CUDA
#include <cuda_runtime_api.h>
#endif
#include <fstream>
#include <iostream>
#include <jsoncpp/json/json.h>
#include <unistd.h>
#include <unordered_map>

namespace gptvm {

#define NV_SPEC_FILE "../config/nv_spec.json"

GVResource GVResource::instance;

const std::vector<GVDevice> &
GVResource::getDevices(GVDeviceType dev_type) const {
  if (devices_map.find(dev_type) == devices_map.end()) {
    static std::vector<GVDevice> emptyDevices;
    return emptyDevices;
  }
  return devices_map.at(dev_type);
}

uint64_t GVResource::getComputeCapability(GVDevice dev,
                                          std::string data_type) const {
  if (dev.device_type == GVDeviceType::GV_CPU) {
    dev = GVDevice(dev.node_id, dev.device_type, 0);
  }
  if (compute_cap.find(dev) == compute_cap.end()) {
    return 0;
  }
  return compute_cap.at(dev).at(data_type);
}

const std::unordered_map<std::string, float>
GVResource::getComputeCapability(GVDevice dev) const {
  if (dev.device_type == GVDeviceType::GV_CPU) {
    dev = GVDevice(dev.node_id, dev.device_type, 0);
  }
  if (compute_cap.find(dev) == compute_cap.end()) {
    return {};
  }
  return compute_cap.at(dev);
}

uint64_t GVResource::getMemorySize(GVDevice dev) const {
  if (dev.device_type == GVDeviceType::GV_CPU) {
    dev = GVDevice(dev.node_id, dev.device_type, 0);
  }
  if (memory_size_map.find(dev) == memory_size_map.end()) {
    return 0;
  }
  return memory_size_map.at(dev);
}

void GVResource::resourceInit() {

  // ############## first ################
  // get the cpu resource of the node
  // set node id, need to find the unique id in the cluster
  node_id = 0;
  // get the cpu number of the node
  uint8_t cpu_num = sysconf(_SC_NPROCESSORS_ONLN);
  std::vector<GVDevice> cpu_devices;
  for (uint8_t i = 0; i < cpu_num; i++) {
    // GVDeviceType.GV_CPU, node_id, i
    GVDevice cpu_device(node_id, GVDeviceType::GV_CPU, i);
    devices_map[GVDeviceType::GV_CPU].push_back(cpu_device);
  }
  // get the memory size of the node, always with device id 0
  GVDevice cpu_device(node_id, GVDeviceType::GV_CPU, 0);
  memory_size_map[cpu_device] = sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGESIZE);

#if GV_HAS_CUDA
  // ############## second ################
  // get the gpu resource of the node
  // get the gpu number of the node
  // first check if there is gpu on the node
  int gpu_num = 0;
  cudaGetDeviceCount(&gpu_num);
  if (gpu_num == 0) {
    return;
  }
  // get the gpu capability config from json file
  std::map<std::string, float> gpu_cap_map;
  // open the json file and import the config
  std::string nv_spec_file =
      GVRuntime::getRuntime().getRuntimePath() + "/" + NV_SPEC_FILE;
  // check if the file exists and open it
  std::ifstream ifs(nv_spec_file);
  if (!ifs.is_open()) {
    std::cout << "Cannot open the file: " << nv_spec_file << std::endl;
    return;
  }
  // read the file
  Json::Reader reader;
  Json::Value root;
  if (!reader.parse(ifs, root)) {
    std::cout << "Failed to parse the file: " << nv_spec_file << std::endl;
    return;
  }
  // get the gpu capability config
  // a: check if the item NV_GPU exists, if it is get the config
  if (!root.isMember("NV_GPU")) {
    std::cout << "Cannot find the item NV_GPU in the file: " << nv_spec_file
              << std::endl;
    return;
  }
  Json::Value nv_gpu = root["NV_GPU"];
  Json::Value cap_map;

  for (uint8_t i = 0; i < gpu_num; i++) {
    // GVDeviceType.GV_NV_GPU, node_id, i
    GVDevice gpu_device(node_id, GVDeviceType::GV_NV_GPU, i);
    devices_map[GV_NV_GPU] = {gpu_device};
    // get the memory size of the gpu
    size_t free, total;
    cudaSetDevice(i);
    cudaMemGetInfo(&free, &total);
    memory_size_map[gpu_device] = total;
    // get the name of the gpu
    char name[128];
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    strcpy(name, prop.name);
    // check if the name contains the key words of gpu_cap_map by iterating
    // every item of nv_gpu
    for (auto i = 0; i < nv_gpu.size(); i++) {
      Json::Value::Members gpu_names = nv_gpu[i].getMemberNames();
      for (auto it = gpu_names.begin(); it != gpu_names.end(); it++) {
        std::string key = *it;
        if (strstr(name, key.c_str()) != NULL) {
          // get the capability map
          cap_map = nv_gpu[i][key];
          break;
        }
      }
    }
    // get the capability of the gpu from cap_map and populate the compute_cap
    Json::Value::Members cap_names = cap_map.getMemberNames();
    for (auto it = cap_names.begin(); it != cap_names.end(); it++) {
      std::string key = *it;
      float value = cap_map[key].asFloat();
      compute_cap[gpu_device][key] = value;
    }
  }
#endif
}

} // namespace gptvm
