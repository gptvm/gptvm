import torch
import argparse
import tensorrt as trt
import pycuda.driver as cuda
import struct
import pytest
from urllib import parse
import numpy
import onnx
import random
import os
import cv2


def lazy_import(name, as_=None):
    # Doesn't handle error_msg well yet
    import importlib

    mod = importlib.import_module(name)
    if as_ is not None:
        name = as_
    # yuck...
    globals()[name] = mod


def gptvm_lazy_import():
    lazy_import("libgptvm")
    lazy_import("gptvm")


def gptvm_inference(model_path, number=None):
    data_dir = model_path + "/"
    model_path = data_dir + "mnist.onnx"

    # parse the model to get the input and output names
    model = onnx.load(model_path)
    input_name = model.graph.input[0].name
    output_name = model.graph.output[0].name
    # get the shape of input and output
    input_shape = model.graph.input[0].type.tensor_type.shape.dim
    input_shape = [dim.dim_value for dim in input_shape]
    output_shape = model.graph.output[0].type.tensor_type.shape.dim
    output_shape = [dim.dim_value for dim in output_shape]
    # get the data type of input and output
    input_type = model.graph.input[0].type.tensor_type.elem_type
    output_type = model.graph.output[0].type.tensor_type.elem_type
    # convert the input_type and output_type to size
    input_type_size = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[input_type].itemsize
    output_type_size = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[output_type].itemsize
    print(
        f"input_name: {input_name}, input_shape: {input_shape}, input_type: {input_type}, input_type_size: {input_type_size}"
        ""
    )
    print(
        f"output_name: {output_name}, output_shape: {output_shape}, output_type: {output_type}, output_type_size: {output_type_size}"
        ""
    )
    # create the input and output tensors
    if number is not None:
        random_number = number
    else:
        # fist read a random image from the mnist dataset
        # get a random number from 0 to 9
        random_number = random.randint(0, 9)
    random_image = data_dir + f"{random_number}.pgm"
    print(f"random_image: {random_image}")
    # read the image data to a numpy array with shape as input_shape
    image = cv2.imread(random_image, cv2.IMREAD_GRAYSCALE)
    print(f"image.shape: {image.shape}, image.dtype: {image.dtype},")
    image = cv2.resize(image, (input_shape[2], input_shape[3]))
    image = image.reshape(input_shape[2], input_shape[3])
    image = image.astype(numpy.float32)
    # convert each item to (1.0 - float(item / 255.0))
    image = 1.0 - image / 255.0
    # convert the numpy array to bytes
    image_bytes = image.tobytes()
    # create the named input object
    input_object = gptvm.GVObject.create(image_bytes, input_shape)
    named_input_object = {input_name: input_object}

    # create task
    task = gptvm.GVTask.create(
        data_dir + "mnist.onnx", gptvm.GV_NV_GPU)
    # launch the task
    outputs = task.launch(named_input_object)
    outputs = next(iter(outputs.get().values())).data
    # convert the bytes to numpy array
    outputs = numpy.frombuffer(outputs, dtype=numpy.float32).reshape(output_shape)
    print(outputs)
    # calculate the softmax of the output
    outputs = numpy.exp(outputs) / numpy.sum(numpy.exp(outputs))
    # get the index of the max value
    index = numpy.argmax(outputs)
    print(f"the predicted number is: {index}")
    return index


def trt_inference():
    import pycuda.autoinit

    data_dir = "./mnist/"
    model_path = data_dir + "mnist.onnx"

    # parse the model to get the input and output names
    model = onnx.load(model_path)
    input_name = model.graph.input[0].name
    output_name = model.graph.output[0].name
    # get the shape of input and output
    input_shape = model.graph.input[0].type.tensor_type.shape.dim
    input_shape = [dim.dim_value for dim in input_shape]
    output_shape = model.graph.output[0].type.tensor_type.shape.dim
    output_shape = [dim.dim_value for dim in output_shape]
    # get the data type of input and output
    input_type = model.graph.input[0].type.tensor_type.elem_type
    output_type = model.graph.output[0].type.tensor_type.elem_type
    # convert the input_type and output_type to size
    input_type_size = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[input_type].itemsize
    output_type_size = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[output_type].itemsize
    print(
        f"input_name: {input_name}, input_shape: {input_shape}, input_type: {input_type}, input_type_size: {input_type_size}"
        ""
    )
    print(
        f"output_name: {output_name}, output_shape: {output_shape}, output_type: {output_type}, output_type_size: {output_type_size}"
        ""
    )
    # create the input and output tensors
    # fist read a random image from the mnist dataset
    # get a random number from 0 to 9
    random_number = random.randint(0, 9)
    random_image = data_dir + f"{random_number}.pgm"
    print(f"random_image: {random_image}")
    # read the image data to a numpy array with shape as input_shape
    image = cv2.imread(random_image, cv2.IMREAD_GRAYSCALE)
    print(f"image.shape: {image.shape}, image.dtype: {image.dtype},")
    image = cv2.resize(image, (input_shape[2], input_shape[3]))
    image = image.reshape(input_shape[2], input_shape[3])
    image = image.astype(numpy.float32)
    # convert each item to (1.0 - float(item / 255.0))
    image = 1.0 - image / 255.0

    # create the inference engine using tensorrt
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    trt_builder = trt.Builder(TRT_LOGGER)
    trt_network = trt_builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    trt_parser = trt.OnnxParser(trt_network, TRT_LOGGER)
    trt_parser.parse_from_file(model_path)
    for i in range(trt_parser.num_errors):
        print(trt_parser.get_error(i))
    trt_config = trt_builder.create_builder_config()
    # trt_config.max_workspace_size = 1 << 30
    serial_engine = trt_builder.build_serialized_network(trt_network, trt_config)
    trt_runtime = trt.Runtime(TRT_LOGGER)
    trt_engine = trt_runtime.deserialize_cuda_engine(serial_engine)
    trt_context = trt_engine.create_execution_context()
    # allocate host memory for input and output
    host_trt_input = numpy.empty(input_shape, dtype=numpy.float32)
    host_trt_output = numpy.empty(output_shape, dtype=numpy.float32)
    # copy the input to the allocated memory
    host_trt_input[0][0] = image
    # create cuda stream
    # stream = cuda.Stream()
    # allocate device memory for input and output
    trt_input = cuda.mem_alloc(host_trt_input.nbytes)
    trt_output = cuda.mem_alloc(host_trt_output.nbytes)
    # copy the input to the device memory
    cuda.memcpy_htod(trt_input, host_trt_input)

    # call the inference
    # trt_context.execute_async_v2(bindings=[int(trt_input), int(trt_output)], stream_handle=stream.handle)
    trt_context.execute_v2(bindings=[int(trt_input), int(trt_output)])
    # copy the output to the host memory
    cuda.memcpy_dtoh(host_trt_output, trt_output)
    # synchronize the stream
    # stream.synchronize()
    print(host_trt_output)
    # calculate the softmax of the output
    host_trt_output = numpy.exp(host_trt_output) / numpy.sum(numpy.exp(host_trt_output))
    # get the index of the max value
    index = numpy.argmax(host_trt_output)
    print(f"the predicted number is: {index}")


def test_mnist():
    gptvm_lazy_import()
    from utils import model_util

    # get current dir
    current_dir = os.getcwd()
    model_name = "mnist"
    model_dir = current_dir + "/model_data"
    # generate a random number to select a image
    random_number = random.randint(0, 9)
    # download the mnist model
    model_util.dload_model("mnist", model_dir)
    # run the inference
    ret = gptvm_inference(model_dir + "/" + model_name, random_number)
    assert ret == random_number


# script selftest
if __name__ == "__main__":
    gptvm_lazy_import()
    # define the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--trt", type=bool, default=False, help="use tensorrt or not")
    args = parser.parse_args()
    if args.trt == True:
        trt_inference()
    else:
        gptvm_inference()
