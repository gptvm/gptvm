#include "tensorrt.h"
#include <cstdlib>

namespace gptvm {

extern "C" {
/// function to create the backend, need to be implemented by each backend
/// and registered in the backend manager
gptvm::GVTensorRT *add_backend(void) { return new GVTensorRT(); }
}

class Logger : public ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    // suppress verbose-level messages
    if (severity <= Severity::kINFO)
      std::cout << msg << std::endl;
  }
} logger;

/// convert the data type from tensorrt to onnx
/// \param tensorrt_type: the data type of tensorrt
/// \return the data type of onnx
size_t tensorrtTypeToSize(nvinfer1::DataType tensorrt_type) {
  switch (tensorrt_type) {
  case nvinfer1::DataType::kINT32:
    return 4;
  case nvinfer1::DataType::kFLOAT:
    return 4;
  case nvinfer1::DataType::kHALF:
    return 2;
  case nvinfer1::DataType::kBOOL:
  case nvinfer1::DataType::kUINT8:
  case nvinfer1::DataType::kINT8:
  case nvinfer1::DataType::kFP8:
    return 1;
  default:
    assert(false);
  }
  return 0;
}
/// \input  model_file: the path of the model file
/// \input  device_type: the device type of the model
/// \output model_info: the model info of the model
GVModel GVTensorRT::build(char *model_data, size_t model_size,
                          const GVNamedShape &input_shapes,
                          GVDeviceType device_type) {
  // initialize a GVModel with input from task_model_info
  GVModel model_info = GVModel();
  // set model file
  this->params.model_data = model_data;
  this->params.model_size = model_size;

  // create builder
  auto builder =
      std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
  if (!builder) {
    std::cout << "create builder failed" << std::endl;
    return model_info;
  }
  // create network
  const auto explicitBatch =
      1U << static_cast<uint32_t>(
          NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
      builder->createNetworkV2(explicitBatch));
  if (!network) {
    std::cout << "create network failed" << std::endl;
    return model_info;
  }
  // create config
  auto config =
      std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  if (!config) {
    std::cout << "create config failed" << std::endl;
    return model_info;
  }
  // create parser
  auto parser = std::unique_ptr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network, logger));
  if (!parser) {
    std::cout << "create parser failed" << std::endl;
    return model_info;
  }
  // contstruct network
  if (!constructNetwork(builder, network, config, parser)) {
    std::cout << "construct network failed" << std::endl;
    return model_info;
  }

  // if the model's input is dynamic, we need to set the optimization profile
  // check if the input is dynamic
  bool is_dynamic = false;
  for (auto input_name : params.input_tensor_names) {
    auto input_dims = params.input_dims[input_name];
    for (auto i = 0; i < input_dims.nbDims; i++) {
      if (input_dims.d[i] == -1) {
        is_dynamic = true;
        break;
      }
    }
  }
  if (is_dynamic) {
    // set the shape of the input and output according to the inputs's shape
    auto profile = builder->createOptimizationProfile();
    // loop through the input tensor names then get the dims
    for (auto shape : input_shapes) {
      nvinfer1::Dims dims_min, dims_opt, dims_max;
      // get the input name
      auto input_name = shape.first;
      // get the input dims
      auto input_dims = shape.second;
      // get the length of the input dims
      auto input_dims_length = input_dims.size();
      dims_min.nbDims = input_dims_length;
      dims_opt.nbDims = input_dims_length;
      dims_max.nbDims = input_dims_length;
      // find the dynamic dimensions accouding to the dims in params,
      // if the dims in params is -1, then we set a suitable value to a new
      // dims, and push it into the profile inputs dims
      for (auto i = 0; i < input_dims_length; i++) {
        if (params.input_dims[input_name].d[i] == -1) {
          auto dim_value = input_dims[i];
          // get the suitable value, Min,Opt,Max, the Opt value is the dim_value
          // the Min value is the Opt value / 2, the Max value is the Opt value
          // + 20
          auto opt_value = dim_value;
          auto min_value = opt_value / 2;
          auto max_value = opt_value + 20;
          dims_min.d[i] = min_value;
          dims_opt.d[i] = opt_value;
          dims_max.d[i] = max_value;
        } else {
          dims_min.d[i] = params.input_dims[input_name].d[i];
          dims_opt.d[i] = params.input_dims[input_name].d[i];
          dims_max.d[i] = params.input_dims[input_name].d[i];
        }
      }
      profile->setDimensions(input_name.c_str(), OptProfileSelector::kMIN,
                             dims_min);
      profile->setDimensions(input_name.c_str(), OptProfileSelector::kOPT,
                             dims_opt);
      profile->setDimensions(input_name.c_str(), OptProfileSelector::kMAX,
                             dims_max);
    }
    config->addOptimizationProfile(profile);

    // set the output dims according to the output dims in params
    // set output info into the model info
    auto output_count = network->getNbOutputs();
    // model_info.outputs_info.resize(output_count);
    for (auto i = 0; i < output_count; i++) {
      // get the shape of the first input, we get the batch and sequence length
      // there
      auto input_dims = input_shapes.at(params.input_tensor_names[0]);
      auto name = network->getOutput(i)->getName();
      GVTensorInfo output_info;
      output_info.elem_size =
          tensorrtTypeToSize(network->getOutput(i)->getType());
      // put the dims into the model info
      for (auto j = 0; j < network->getOutput(i)->getDimensions().nbDims; j++) {
        // get the dim value
        auto dim_value = network->getOutput(i)->getDimensions().d[j];
        // if the dim value is -1, it is a dynamic input
        if (dim_value == -1) {
          output_info.shape.push_back(input_dims[j]);
        } else {
          output_info.shape.push_back(dim_value);
        }
      }
      model_info.outputs_info[name] = output_info;
    }
  }

  // config->setFlag(nvinfer1::BuilderFlag::kFP16);
  std::unique_ptr<IHostMemory> plan{
      builder->buildSerializedNetwork(*network, *config)};
  if (!plan) {
    std::cout << "build serialized network failed" << std::endl;
    return model_info;
  }

  // create runtime
  m_runtime =
      std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
  if (!m_runtime) {
    std::cout << "create runtime failed" << std::endl;
    return model_info;
  }

  // deserialize the engine
  m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(
      m_runtime->deserializeCudaEngine(plan->data(), plan->size()));
  if (!m_engine) {
    std::cout << "deserialize cuda engine failed" << std::endl;
    return model_info;
  }

  // set the handler
  model_info.handler = static_cast<void *>(m_engine.get());

  // populate the model info with the input and output info
  auto input_count = network->getNbInputs();
  auto output_count = network->getNbOutputs();
  // put the input shape from the params into the model info
  for (auto i = 0; i < input_count; i++) {
    auto input_name = params.input_tensor_names[i];
    model_info.inputs_info[input_name].elem_size =
        tensorrtTypeToSize(network->getInput(i)->getType());
    for (auto j = 0; j < params.input_dims[input_name].nbDims; j++) {
      model_info.inputs_info[input_name].shape.push_back(
          params.input_dims[input_name].d[j]);
    }
  }
  // put the output shape from the params into the model info
  if (!is_dynamic) {
    for (auto i = 0; i < output_count; i++) {
      auto output_name = params.output_tensor_names[i];
      model_info.outputs_info[output_name].elem_size =
          tensorrtTypeToSize(network->getOutput(i)->getType());
      for (auto j = 0; j < params.output_dims[output_name].nbDims; j++) {
        model_info.outputs_info[output_name].shape.push_back(
            params.output_dims[output_name].d[j]);
      }
    }
  }
  return model_info;
}

GVNamedOutput GVTensorRT::run(GVModel model_info,
                              const GVNamedShape &input_shapes,
                              const GVNamedBuffer &inputs, GVDevice device) {
  // create cuda stream
  cudaStreamCreate(&stream);

  //  create execution context
  auto context = std::shared_ptr<nvinfer1::IExecutionContext>(
      m_engine->createExecutionContext());
  assert(context);

  // arrange the data into a vector
  std::vector<void *> datas(params.output_tensor_names.size() +
                            params.input_tensor_names.size());
  for (auto input_name : params.input_tensor_names) {
    int i = m_engine->getBindingIndex(input_name.c_str());
    datas[i] = inputs.find(input_name)->second.getData();
  }

  // if the model's input is dynamic, we need to set the optimization profile
  // check if the input is dynamic
  bool is_dynamic = false;
  for (auto input_name : params.input_tensor_names) {
    auto input_dims = params.input_dims[input_name];
    for (auto i = 0; i < input_dims.nbDims; i++) {
      if (input_dims.d[i] == -1) {
        is_dynamic = true;
        break;
      }
    }
  }
  if (is_dynamic) {
    // TODO: set the binding dimensions according to the model info
    context->setOptimizationProfileAsync(0, stream);
    for (auto x : inputs) {
      // get the input name
      auto input_name = x.first;
      // get the input dims
      auto input_dims = input_shapes.at(input_name);
      // get the length of the input dims
      auto input_dims_length = input_dims.size();
      // construct the dims according to the input shape
      nvinfer1::Dims input_dims_trt = params.input_dims[input_name];
      for (auto j = 0; j < input_dims_length; j++) {
        input_dims_trt.d[j] = input_dims[j];
      }
      // set the input shape
      if (!context->setInputShape(input_name.c_str(), input_dims_trt)) {
        std::cout << "set input shape failed for " << input_name << std::endl;
        assert(false);
      }
    }
  }

  GVNamedBuffer outputs;
  GVNamedOutput outputs_cpu;
  for (auto name : params.output_tensor_names) {
    auto dims = context->getTensorShape(name.c_str());
    size_t size = model_info.outputs_info[name].elem_size;
    std::vector<int64_t> shape;
    for (int i = 0; i < dims.nbDims; i++) {
      size *= dims.d[i];
      shape.push_back(dims.d[i]);
    }
    auto data = memAlloc(size, device);
    assert(data);
    outputs[name] = GVBuffer(data, size, device);
    auto data_cpu = new char[size];
    assert(data_cpu);
    outputs_cpu[name] = std::make_tuple((int64_t)data_cpu, size, shape);
    int i = m_engine->getBindingIndex(name.c_str());
    datas[i] = outputs.find(name)->second.getData();
  }
  // run the inference
  assert(context->executeV2(datas.data()));

  for (auto &[name, output] : outputs) {
    auto size = output.impl->size;
    auto data = (char *)std::get<0>(outputs_cpu[name]);
    auto ret = memCopy(data, output.impl->data, size, GV_MEMCPY_D2H, device);
    assert(ret == size);
  }

  for (auto &[name, buffer] : outputs) {
    memFree(buffer.impl->data, device);
    buffer.impl->size = 0;
  }

  return outputs_cpu;
}

/// \brief get the device list
/// \return the device list
std::vector<GVDevice> GVTensorRT::getDeviceList(void) {
  // find the GPU devices and return the list
  std::vector<GVDevice> list;
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  char *env = getenv("CUDA_VISIBLE_DEVICES");
  int gpu_id = 0;
  if (env && device_count) {
    assert(device_count == 1);
    gpu_id = atoi(env);
  }

  for (int i = 0; i < device_count; ++i) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    if (prop.major >= 2) {
      // For now we take node id by 0, will change it later
      list.push_back(GVDevice(0, GV_NV_GPU, gpu_id + i));
    }
  }

  return list;
}

/// \brief construct the network, will take care of backend specific
///        parameters in the future; work with backend config
/// \param builder: the builder
/// \param network: the network
/// \param config: the config
/// \param parser: the parser
bool GVTensorRT::constructNetwork(
    std::unique_ptr<nvinfer1::IBuilder> &builder,
    std::unique_ptr<nvinfer1::INetworkDefinition> &network,
    std::unique_ptr<nvinfer1::IBuilderConfig> &config,
    std::unique_ptr<nvonnxparser::IParser> &parser) {

  // parse the model to populate the network, then set the outputs
  auto parsed = parser->parse(params.model_data, params.model_size);
  if (!parsed) {
    for (int i = 0; i < parser->getNbErrors(); i++)
      std::cout << "TensorRT parser errors:" << parser->getError(0)->desc()
                << std::endl;
    return false;
  }

  // fill up the input and output tensor names and dims
  for (auto i = 0; i < network->getNbInputs(); i++) {
    params.input_tensor_names.push_back(network->getInput(i)->getName());
    params.input_dims[network->getInput(i)->getName()] =
        network->getInput(i)->getDimensions();
  }
  for (auto i = 0; i < network->getNbOutputs(); i++) {
    params.output_tensor_names.push_back(network->getOutput(i)->getName());
    params.output_dims[network->getOutput(i)->getName()] =
        network->getOutput(i)->getDimensions();
  }

  // TODO, take care of all of the params
  return true;
}

} // namespace gptvm
