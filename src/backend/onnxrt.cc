#include "gptvm/backend/backend.h"
#include "gptvm/runtime/buffer.h"

#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

#include <unistd.h>

// template <typename T>
// std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) {
//   os << "[";
//   for (int i = 0; i < v.size(); ++i) {
//     os << v[i];
//     if (i != v.size() - 1) {
//       os << ", ";
//     }
//   }
//   os << "]";
//   return os;
// }

static int32_t onnxTypeToSize(const ONNXTensorElementDataType &type) {
  switch (type) {
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    return 1;
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
    return 2;
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
    return 4;
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
    return 8;
  default:
    assert(false);
  }

  return 0;
}

namespace gptvm {

class GVOnnx : public GVBackend {
public:
  virtual GVModel build(char *model_data, size_t model_size,
                        const GVNamedShape &input_shapes,
                        GVDeviceType device_type) override;
  virtual GVNamedOutput run(GVModel model_info,
                            const GVNamedShape &input_shapes,
                            const GVNamedBuffer &inputs,
                            GVDevice device) override;
  virtual std::vector<GVDevice> getDeviceList(void) override;

private:
  Ort::AllocatorWithDefaultOptions allocator;
};

extern "C" {
GVBackend *add_backend(void) { return new GVOnnx(); }
}

std::vector<GVDevice> GVOnnx::getDeviceList() {
  std::vector<GVDevice> devices;
  devices.push_back(GVDevice(0, GV_CPU, 0));

  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count) {
    char *env = getenv("CUDA_VISIBLE_DEVICES");
    int gpu_id = 0;
    if (env) {
      assert(device_count == 1);
      gpu_id = atoi(env);
    }

    for (int i = 0; i < device_count; ++i) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);
      if (prop.major >= 2) {
        devices.push_back(GVDevice(0, GV_NV_GPU, gpu_id + i));
      }
    }
  }
  return devices;
}

template <typename T> T vectorProduct(const std::vector<T> &v) {
  return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

static Ort::Value createTensorByType(const ONNXTensorElementDataType &type,
                                     Ort::MemoryInfo &memoryInfo, char *data,
                                     std::vector<int64_t> &shape) {
  switch (type) {
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    return Ort::Value::CreateTensor<float>(memoryInfo, (float *)data,
                                           vectorProduct(shape), shape.data(),
                                           shape.size());

  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    return Ort::Value::CreateTensor<uint8_t>(memoryInfo, (uint8_t *)data,
                                             vectorProduct(shape), shape.data(),
                                             shape.size());
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    return Ort::Value::CreateTensor<int8_t>(memoryInfo, (int8_t *)data,
                                            vectorProduct(shape), shape.data(),
                                            shape.size());
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    return Ort::Value::CreateTensor<uint16_t>(memoryInfo, (uint16_t *)data,
                                              vectorProduct(shape),
                                              shape.data(), shape.size());
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    return Ort::Value::CreateTensor<int16_t>(memoryInfo, (int16_t *)data,
                                             vectorProduct(shape), shape.data(),
                                             shape.size());
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    return Ort::Value::CreateTensor<int32_t>(memoryInfo, (int32_t *)data,
                                             vectorProduct(shape), shape.data(),
                                             shape.size());
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    return Ort::Value::CreateTensor<int64_t>(memoryInfo, (int64_t *)data,
                                             vectorProduct(shape), shape.data(),
                                             shape.size());

  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
    return Ort::Value::CreateTensor<bool>(memoryInfo, (bool *)data,
                                          vectorProduct(shape), shape.data(),
                                          shape.size());
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    return Ort::Value::CreateTensor<double>(memoryInfo, (double *)data,
                                            vectorProduct(shape), shape.data(),
                                            shape.size());
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
    return Ort::Value::CreateTensor<uint32_t>(memoryInfo, (uint32_t *)data,
                                              vectorProduct(shape),
                                              shape.data(), shape.size());
  case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
    return Ort::Value::CreateTensor<uint64_t>(memoryInfo, (uint64_t *)data,
                                              vectorProduct(shape),
                                              shape.data(), shape.size());
  default:
    assert(false);
    break;
  }
  assert(false);
  return Ort::Value(nullptr);
}

GVModel GVOnnx::build(char *model_data, size_t model_size,
                      const GVNamedShape &input_shapes,
                      GVDeviceType device_type) {
  GVModel model_info;
  Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ORB");
  Ort::SessionOptions sessionOptions;
  sessionOptions.SetIntraOpNumThreads(1);
  if (device_type == GV_NV_GPU)
    OrtStatus *status =
        OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
  auto session = new Ort::Session(env, model_data, model_size, sessionOptions);
  model_info.handler = session;

  size_t input_count = session->GetInputCount();
  size_t output_count = session->GetOutputCount();

  for (auto i = 0; i < input_count; i++) {
    Ort::TypeInfo inputTypeInfo = session->GetInputTypeInfo(i);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

    auto input_name =
        std::string(session->GetInputNameAllocated(i, allocator).get());
    model_info.inputs_info[input_name].elem_size =
        onnxTypeToSize(inputTensorInfo.GetElementType());
    model_info.inputs_info[input_name].shape = inputTensorInfo.GetShape();
  }

  for (auto i = 0; i < output_count; i++) {
    Ort::TypeInfo outputTypeInfo = session->GetOutputTypeInfo(i);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

    auto output_name =
        std::string(session->GetOutputNameAllocated(i, allocator).get());
    model_info.outputs_info[output_name].elem_size =
        onnxTypeToSize(outputTensorInfo.GetElementType());
    model_info.outputs_info[output_name].shape = outputTensorInfo.GetShape();
  }

  return model_info;
}

GVNamedOutput GVOnnx::run(GVModel model_info, const GVNamedShape &input_shapes,
                          const GVNamedBuffer &inputs, GVDevice device) {

  Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "ORB");
  std::vector<Ort::Value> input_tensors;
  Ort::MemoryInfo *memoryInfo;

  if (device.device_type == GV_NV_GPU)
    memoryInfo = new Ort::MemoryInfo(
        "Cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);
  else if (device.device_type == GV_CPU)
    memoryInfo = new Ort::MemoryInfo("Cpu", OrtAllocatorType::OrtArenaAllocator,
                                     0, OrtMemTypeDefault);
  else
    assert(false);

  std::vector<const char *> inputs_name;
  for (auto i = 0; i < inputs.size(); i++) {
    Ort::TypeInfo inputTypeInfo =
        ((Ort::Session *)model_info.handler)->GetInputTypeInfo(i);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    auto input_name = std::string(((Ort::Session *)model_info.handler)
                                      ->GetInputNameAllocated(i, allocator)
                                      .get());
    auto shape = input_shapes.find(input_name)->second;
    auto input_tensor =
        createTensorByType(inputTensorInfo.GetElementType(), *memoryInfo,
                           inputs.find(input_name)->second.getData(), shape);
    input_tensors.push_back(std::move(input_tensor));
    inputs_name.push_back(inputs.find(input_name)->first.c_str());
  }

  std::vector<const char *> outputs_name;
  for (auto &[name, _] : model_info.outputs_info)
    outputs_name.push_back(name.c_str());

  // run the inference
  auto outputs_tensor = ((Ort::Session *)model_info.handler)
                            ->Run(Ort::RunOptions{nullptr}, inputs_name.data(),
                                  input_tensors.data(), inputs_name.size(),
                                  outputs_name.data(), outputs_name.size());
  GVNamedOutput outputs_host;

  if (outputs_tensor.size() > 0) {
    for (int i = 0; i < outputs_tensor.size(); i++) {
      auto &output = outputs_tensor[i];
      assert(output.IsTensor());
      auto name = outputs_name[i];

      auto output_info = output.GetTensorTypeAndShapeInfo();
      auto type = output_info.GetElementType();
      auto shape = output_info.GetShape();

      auto size = onnxTypeToSize(type) * vectorProduct(shape);
      auto data_host = new char[size];
      assert(data_host);
      auto data_devcie = output.GetTensorMutableData<char>();
      // FIXME: ORT run return CPU memory buffer.
      auto ret = memCopy(data_host, (char *)data_devcie, size, GV_MEMCPY_D2H,
                         GVDevice(0));
      assert(ret == size);
      outputs_host[name] = std::make_tuple((int64_t)data_host, size, shape);
    }
  }

  // TODO: How to keep lifetime of output tensor?
  return outputs_host;
}

} // namespace gptvm
