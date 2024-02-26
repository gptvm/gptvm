#include "backend/tensorrt.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace gptvm;

/// \brief  this is a dummy function to test the backend of tensorrt

int main(int argc, char **argv) {
  // get the data directory from the command line
  std::string data_dir = "./";
  if (argc == 2) {
    std::cout << "Using data directory: " << argv[1] << std::endl;
    data_dir = argv[1];
  } else if (argc == 1) {
    std::cout << "Using current dir as data directory!" << std::endl;
  }
  std::string model_file = data_dir + "mnist.onnx";

#if 0
  // here dlopen the libob_tensorrt.so
  // and get the function pointer of add_backend
  // and create the GVTensorRT object
  std::string backend_name = "libob_tensorrt.so";
  void *handle = dlopen(backend_name.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (!handle) {
    std::cout << "dlopen " << backend_name << " failed" << std::endl;
    return -1;
  }
  typedef GVTensorRT *(*add_backend_t)();
  add_backend_t add_backend = (add_backend_t)dlsym(handle, "add_backend");
  if (!add_backend) {
    std::cout << "dlsym add_backend failed" << std::endl;
    return -1;
  }
  //GVBackend *tensorrt = add_backend();
  std::shared_ptr<GVTensorRT> tensorrt(add_backend());
  if (!tensorrt) {
    std::cout << "add_backend failed" << std::endl;
    return -1;
  } else {
    std::cout << "add_backend success" << std::endl;
  }
#else
  // create GVTensorRT object
  GVTensorRT *tensorrt = new GVTensorRT();
#endif

  // build model
  auto model_info =
      tensorrt->build(NULL, 0, GVNamedShape(), GVDeviceType::GV_NV_GPU);
  if (!model_info.inputs_info.size() || !model_info.outputs_info.size()) {
    std::cout << "build model failed" << std::endl;
    return -1;
  }

  // get the input and output shape from model_info
  // for mnist.onnx, we only have one input and one output
  // so we can get the shape from the first element of the map
  auto named_input_shape = *model_info.inputs_info.begin();
  auto named_output_shape = *model_info.outputs_info.begin();

  // create input data
  const int inputH = named_input_shape.second.shape[2];
  const int inputW = named_input_shape.second.shape[3];

  // Read a random digit file
  srand(unsigned(time(nullptr)));
  std::vector<uint8_t> fileData(inputH * inputW);
  int mNumber = rand() % 10;
  // read the data from the file
  std::ifstream file(data_dir.c_str() + std::to_string(mNumber) + ".pgm",
                     std::ifstream::binary);
  if (!file.is_open()) {
    std::cout << "read file failed" << std::endl;
    return -1;
  }
  // read the file content to fileData
  std::string magic, w, h, max;
  file >> magic >> w >> h >> max;
  file.seekg(1, file.cur);
  file.read(reinterpret_cast<char *>(fileData.data()), inputH * inputW);

  // Print an ascii representation
  std::cout << "Input:" << std::endl;
  for (int i = 0; i < inputH * inputW; i++) {
    std::cout << (" .:-=+*#%@"[fileData[i] / 26])
              << (((i + 1) % inputW) ? "" : "\n");
  }
  std::cout << std::endl;

  float *hostDataBuffer =
      static_cast<float *>(malloc(inputH * inputW * sizeof(float)));
  for (int i = 0; i < inputH * inputW; i++) {
    hostDataBuffer[i] = 1.0 - float(fileData[i] / 255.0);
  }

  // Create GVBuffer
  auto data = tensorrt->memAlloc(inputH * inputW * sizeof(float),
                                 tensorrt->getDeviceList()[0]);
  // this is tricky, we need to create a GVBuffer object with size = 0,
  // just to avoid the assert in GVBuffer class
  GVBuffer input_buffer(data, 0, tensorrt->getDeviceList()[0]);
  GVNamedBuffer named_input_buffer;
  named_input_buffer[named_input_shape.first] = input_buffer;
  // copy input data to device
  tensorrt->memCopy(data, (char *)hostDataBuffer,
                    inputH * inputW * sizeof(float), GV_MEMCPY_H2D,
                    tensorrt->getDeviceList()[0]);

  // create output buffer
  const int outputSize = named_output_shape.second.shape[1];
  std::vector<float> output(outputSize);
  auto output_data = tensorrt->memAlloc(outputSize * sizeof(float),
                                        tensorrt->getDeviceList()[0]);
  // this is tricky, we need to create a GVBuffer object with size = 0,
  // just to avoid the assert in GVBuffer class
  GVBuffer output_buffer(output_data, 0, tensorrt->getDeviceList()[0]);
  GVNamedBuffer named_output_buffer;
  named_output_buffer[named_output_shape.first] = output_buffer;

  GVNamedShape input_shape;
  input_shape[named_input_shape.first] = named_input_shape.second.shape;

  // run model
  auto ret = tensorrt->run(model_info, input_shape, {named_input_buffer},
                           GVDevice(0, gptvm::GV_NV_GPU, 0));
  if (!ret.size()) {
    std::cout << "run model failed" << std::endl;
    return -1;
  }

  // copy output data to host
  tensorrt->memCopy(
      (char *)output.data(), (char *)std::get<0>(ret.begin()->second),
      outputSize * sizeof(float), GV_MEMCPY_D2H, tensorrt->getDeviceList()[0]);

  // print output
  float val{0.0F};
  int idx{0};

  // Calculate Softmax
  float sum{0.0F};
  for (int i = 0; i < outputSize; i++) {
    output[i] = std::exp(output[i]);
    sum += output[i];
  }

  std::cout << "Output:" << std::endl;
  for (int i = 0; i < outputSize; i++) {
    output[i] /= sum;
    val = std::max(val, output[i]);
    if (val == output[i]) {
      idx = i;
    }

    std::cout << " Prob " << i << "  " << std::fixed << std::setw(5)
              << std::setprecision(4) << output[i] << " "
              << "Class " << i << ": "
              << std::string(int(std::floor(output[i] * 10 + 0.5F)), '*')
              << std::endl;
  }
  std::cout << std::endl;
}
