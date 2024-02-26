#pragma once

#include "gptvm/backend/backend.h"
#include "onnx/onnx_pb.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>

namespace gptvm {

using namespace nvinfer1;

///
/// \brief The GVTRParams structure groups the basic parameters required by
///        the networks.
struct GVTRTParams {
  int32_t batch_size{1}; // Number of inputs in a batch
  int32_t dla_core{-1};  // Specify the DLA core to run network on.
  bool int8{false};      // Allow runnning the network in Int8 mode.
  bool fp16{false};      // Allow running the network in FP16 mode.
  std::vector<std::string>
      data_dirs; // Directory paths where sample data files are stored
  // the seies of the input and output tensor names is accouding
  // network->getInput(i) and network->getOutput(i)
  std::vector<std::string> input_tensor_names;
  std::vector<std::string> output_tensor_names;
  std::unordered_map<std::string, nvinfer1::Dims> input_dims;
  std::unordered_map<std::string, nvinfer1::Dims> output_dims;
  char *model_data;
  size_t model_size;
};

class GVTensorRT : public GVBackend {
public:
  GVTensorRT() = default;
  virtual ~GVTensorRT() = default;

  // method from GVBackend
  virtual std::vector<GVDevice> getDeviceList(void) override;
  virtual GVModel build(char *model_data, size_t model_size,
                        const GVNamedShape &input_shapes,
                        GVDeviceType device_type) override;
  virtual GVNamedOutput run(GVModel model_info,
                            const GVNamedShape &input_shapes,
                            const GVNamedBuffer &inputs,
                            GVDevice device) override;

private:
  GVTRTParams params;
  std::shared_ptr<nvinfer1::IRuntime>
      m_runtime; // The TensorRT runtime used to deserialize the engine
  std::shared_ptr<nvinfer1::ICudaEngine>
      m_engine;        // The TensorRT engine used to run the network
  cudaStream_t stream; // The CUDA stream used to enqueue the inference work

  /// Parses an ONNX model and creates a TensorRT network, handle backend
  /// specific configuration in the future, for now just a placeholder
  bool constructNetwork(std::unique_ptr<nvinfer1::IBuilder> &builder,
                        std::unique_ptr<nvinfer1::INetworkDefinition> &network,
                        std::unique_ptr<nvinfer1::IBuilderConfig> &config,
                        std::unique_ptr<nvonnxparser::IParser> &parser);
};

} // namespace gptvm
