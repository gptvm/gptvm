import gptvm
import torch, torch.fx, onnx
import os, io


onnx_backend = "gptvm_runtime"


def backend(gm: torch.fx.GraphModule, example_inputs):
    gptvm.log.debug(f"Compiling torch fx graph:\n{str(gm)}")

    input_names = []
    dynamic_axes = dict()
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            input_names.append(node.name)
            if "example_value" in node.meta:
                d_axes = []
                for i, s in enumerate(node.meta["example_value"].size()):
                    if isinstance(s, torch.SymInt):
                        d_axes.append(i)
                if d_axes:
                    dynamic_axes[node.name] = d_axes
        elif node.op == "output":
            output_names = [n.name for n in node.args[0]]
    inputs = [-1 if isinstance(i, torch.SymInt) else i for i in example_inputs]

    try:
        model_bytes = io.BytesIO()
        torch.onnx.export(
            gm,
            tuple(inputs),
            model_bytes,
            export_params=False,
            do_constant_folding=False,  # Disable constant folding to preseve original parameter names
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=15,
        )
    except Exception as e:
        print(
            "Failed to export pytorch model to onnx, fallback to pytorch native forward",
            e,
        )
        return gm.forward

    model = onnx.load_model_from_string(model_bytes.getvalue())

    model_inputs = [node.name for node in model.graph.input]
    params = dict()
    for name, param in gm.state_dict().items():
        if name in model_inputs:
            params[name] = param

    if onnx_backend == "onnx_runtime":
        import onnxruntime, numpy

        session = onnxruntime.InferenceSession(model_bytes.getvalue())

        def _run_model(*inputs):
            input_feeds = dict()
            for k, v in params.items():
                input_feeds[k] = v.numpy()
            for k, v in zip(input_names, inputs):
                if isinstance(v, torch.Tensor):
                    input_feeds[k] = v.numpy()
            results = []
            for r in session.run(None, input_feeds):
                if isinstance(r, numpy.ndarray):
                    results.append(torch.from_numpy(r))
                else:
                    results.append(r)
            return results

        return _run_model

    if onnx_backend == "gptvm_runtime":
        import numpy

        task = gptvm.GVTask.create(
            model_bytes.getvalue(), gptvm.GVDeviceType.GV_NV_GPU, "gv_tensorrt"
        )

        def _run_model(*inputs):
            input_feeds = dict()
            for k, v in params.items():
                input_feeds[k] = gptvm.GVObject.create(v)
            for k, v in zip(input_names, inputs):
                if isinstance(v, torch.Tensor):
                    input_feeds[k] = gptvm.GVObject.create(v)
            object = task.launch(input_feeds)
            results = object.get()
            outputs = []
            for name in results:
                outputs.append(torch.Tensor(numpy.frombuffer(results[name].data, dtype=numpy.int32)))
            return [outputs]

        return _run_model

    return gm.forward


def compile(forward):
    gptvm.log.debug(f"Captured torch forward function: {forward}")
    return torch.compile(forward, backend=backend)
