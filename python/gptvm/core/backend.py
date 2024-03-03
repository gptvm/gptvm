from gptvm.core.device import *

import libgptvm

import ray
import os
from ctypes import *
import numpy as np


class GVBackend(object):
    """
    class GVBackend reprensents a backend to run task.

    GVBackend run based on backend of libgptvm module.
    """

    def __init__(self):
        self.__use_cache = False

    def forward(self, task, named_objects, named_shapes):
        """Forward a model including building and running.

        Args:
            task: Task to run on the backend.
            named_objects: Named input objects.
            named_shapes: Shape of named input objects.
            device: Device to run.

        Returns:
            Output object.
        """
        if type(task.device) == GVDeviceType:
            device_type = task.device
            if task.device == GVDeviceType.GV_CPU:
                device = GVDevice([0, task.device, 0])
            elif task.device == GVDeviceType.GV_NV_GPU:
                gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES")
                assert gpu_id != None
                device = GVDevice([0, task.device, int(gpu_id)])
            else:
                assert False
        elif type(task.device) == GVDevice:
            device_type = task.device.device_type
            device = task.device
            # Local backend no need to check node id.
            device.clear_node()
        else:
            assert False

        if self.__use_cache == False:
            model_data = ray.get(task.model_ref)
            model_addr = cast(cast(model_data, POINTER(c_char)), c_void_p).value
            model_size = len(model_data)
            self.__obbackend = libgptvm.GVBackend(
                device.global_id,
                task.backend_name,
                os.path.dirname(libgptvm.__file__) + "/../backend",
                [model_addr, model_size],
                task.model_inputs_shape,
                task.model_inputs_elem_size,
                task.model_outputs_shape,
                task.model_outputs_elem_size,
            )

        inputs = {}
        input_data = {}
        for name in named_objects:
            input_data[name] = ray.get(named_objects[name])
            inputs[name] = [
                cast(cast(input_data[name], POINTER(c_char)), c_void_p).value,
                len(input_data[name]),
            ]

        outputs = self.__obbackend.forward(inputs, named_shapes, self.__use_cache)

        if self.__use_cache == False:
            self.__use_cache = True

        outputs_bytes = {}
        outputs_shapes = {}
        free_list = []
        for name in outputs:
            buffer = (c_char * outputs[name][1]).from_address(outputs[name][0])
            outputs_bytes[name] = bytes(buffer)
            outputs_shapes[name] = outputs[name][2]
            free_list.append((outputs[name][0], device.global_id))

            # debug_output = np.frombuffer(bytes(buffer), dtype=np.float32)
            # debug_output = debug_output.reshape(outputs[name][2])
            # np.save("/tmp/" + name + ".npy", debug_output)
        self.__obbackend.memFree(free_list)

        return outputs_bytes, outputs_shapes, device
