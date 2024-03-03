from gptvm.core.device import *
from gptvm.core.object import *
from gptvm.core.backend import *
from gptvm.core.resource import ClusterResource

import onnx
from functools import singledispatchmethod
import io


class GVTask(object):
    """
    class GVTask represents a chunk of computation to be scheduled.

    An GVTask takes variadic GVObject as arguments, and produces one GVObject
    as result. Application thus can be represented as a task graph, with
    GVTasks as nodes, GVObjects as edges.
    An GVTask can be scheduled to start execution at any time once all its
    arguments are produced and transfered to memory storage the scheduled
    execution unit has access to.
    """

    def __init__(self, model_bytes, device, backend_name, gpu_number=1):
        self.device = device
        self.backend_name = backend_name
        self.model_ref = ray.put(model_bytes)
        self.model_inputs_name = []
        self.model_outputs_name = []
        self.model_inputs_shape = {}
        self.model_inputs_elem_size = {}
        self.model_outputs_shape = {}
        self.model_outputs_elem_size = {}

        if type(device) == GVDeviceType:
            self.backend = ray.remote(
                num_cpus=1 if device == GVDeviceType.GV_CPU else 0,
                num_gpus=gpu_number if device == GVDeviceType.GV_NV_GPU else 0,
            )(GVBackend).remote()
        elif type(device) == GVDevice:
            device_type = device.device_type
            self.backend = ray.remote(
                num_cpus=1 if device_type == GVDeviceType.GV_CPU else 0,
                num_gpus=gpu_number if device_type == GVDeviceType.GV_NV_GPU else 0,
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    ClusterResource.get_ray_id(device.node_id), False
                ),
            )(GVBackend).remote()
        else:
            assert False

    @singledispatchmethod
    def launch(self, args: list | dict):
        """Schedule an execution of the task.

        Args:
            args: List of task arguments in pre-defined argument order.

        Returns:
            GVObject associated with the output of the task.
        """
        assert False, "Unsupport arguments of launch"

    @launch.register
    def _(self, args: list):
        assert len(args), "Empty args."

        inputs = dict(zip(self.model_inputs_name, args))
        return self.launch(inputs)

    @launch.register
    def _(self, args: dict):
        assert len(args), "Empty args."

        inputs = {}
        inputs_shape = {}
        for name, object in args.items():
            inputs[name] = object.getDataRef()
            inputs_shape[name] = object.getShape()

        output = GVObject()
        output.setDataRef(self.func.remote(self, inputs, inputs_shape))
        return output

    @singledispatchmethod
    @staticmethod
    def create(model: str | bytes, device_type: GVDeviceType, backend_name: str):
        """Construct an GVTask from a model file or model bytes.

        Args:
            model: Path or bytes data of model file.
            device_type: Type or ID of device that the model should be scheduled on. Default is GV_CPU.
            backend_name: Name of backend. Default is Empty value which means to use any backend.

        Returns:
            Created GVTask.
        """
        assert False, "Unsupport arguments of create"

    @create.register
    @staticmethod
    def _(
        model_file: str,
        device: GVDeviceType | GVDevice = GVDeviceType.GV_CPU,
        backend_name="",
        gpu_number=1
    ):
        assert type(device) == GVDeviceType or type(device) == GVDevice, "Type error."
        assert type(backend_name) == str, "Type error."
        assert os.path.exists(model_file), "File does not exist."

        with open(model_file, "rb") as file:
            model_bytes = file.read()

        return GVTask.create(model_bytes, device, backend_name, gpu_number)

    @create.register
    @staticmethod
    def _(
        model_bytes: bytes,
        device: GVDeviceType | GVDevice = GVDeviceType.GV_CPU,
        backend_name="",
        gpu_number=1
    ):
        assert type(device) == GVDeviceType or type(device) == GVDevice, "Type error."
        assert type(backend_name) == str, "Type error."
        assert len(model_bytes), "Empty model."

        task = GVTask(model_bytes, device, backend_name, gpu_number)
        model = onnx.load(io.BytesIO(model_bytes))
        assert len(model.graph.input), "Empty input."
        assert len(model.graph.output), "Empty output."

        initializer_names = []
        for initializer in model.graph.initializer:
            initializer_names.append(initializer.name)

        for input in model.graph.input:
            if initializer_names.count(input.name) == 0:
                task.model_inputs_name.append(input.name)

                # TODO: need to check dim_param for dynamic shape
                task.model_inputs_shape[input.name] = [
                    dim.dim_value if dim.dim_value > 0 else -1
                    for dim in input.type.tensor_type.shape.dim
                ]
                task.model_inputs_elem_size[
                    input.name
                ] = onnx.helper.tensor_dtype_to_np_dtype(
                    input.type.tensor_type.elem_type
                ).itemsize

        for output in model.graph.output:
            task.model_outputs_name.append(output.name)
            task.model_outputs_shape[output.name] = [
                dim.dim_value if dim.dim_value > 0 else -1
                for dim in output.type.tensor_type.shape.dim
            ]
            task.model_outputs_elem_size[
                output.name
            ] = onnx.helper.tensor_dtype_to_np_dtype(
                output.type.tensor_type.elem_type
            ).itemsize

        task.func = task.backend.forward
        return task
