from transformers import FillMaskPipeline, AutoTokenizer, AutoModelForMaskedLM

# from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel

import torch
import struct
from urllib import parse
import numpy
import os
import ray


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


class OnnxFillMaskPipeline(FillMaskPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """ 
        # ****this is for onnxruntime****  
        options = SessionOptions()
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = InferenceSession("onnx/model.onnx", sess_options=options, providers=["TensorrtExecutionProvider"])
        self.session.disable_fallback()
        """
        # ****this is for gptvm****
        self.task = gptvm.GVTask.create("model_data/bert/model.onnx", gptvm.GV_NV_GPU)

    def _forward(self, model_inputs):
        print(model_inputs)
        # ****this is for native transformers****
        # model_outputs = self.model(**model_inputs)
        """
        # ****this is for onnxruntime****
        xinputs = {k: v.numpy() for k, v in model_inputs.items()}
        outputs_name = list(map(lambda x: x.name, self.session.get_outputs()))
        outputs = self.session.run(output_names=outputs_name, input_feed=xinputs)
        outputs = torch.tensor(outputs[0])
        """
        #  ****this is for gptvm*****
        # model input like below:
        # {'input_ids': tensor([[ 101, 3000, 2003, 1996,  103, 1997, 2605, 1012,  102]]),
        #  'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0]]),
        #  'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]])}

        # get the dims of input_ids
        dims = list(model_inputs["input_ids"].shape)

        # pack each item of model input to bytes
        objects = {}
        for key, value in model_inputs.items():
            array = torch.tensor(value).numpy().astype(numpy.int64)
            array_bytes = array.tobytes()
            lobject = gptvm.GVObject.create(array_bytes, list(value.shape))
            objects[key] = lobject

        # call named launch function of gptvm
        outputs = self.task.launch(objects)
        outputs = next(iter(outputs.get().values())).data
        # hard code the dims of output_ids
        outputs = numpy.frombuffer(
            outputs, dtype=numpy.float32, count=dims[0] * dims[1] * 30522
        ).reshape(dims[0], dims[1], 30522)
        outputs = torch.tensor(outputs)

        model_outputs = {"logits": outputs}
        model_outputs["input_ids"] = model_inputs["input_ids"]

        return model_outputs


def pipeline(backend, model):
    pipelines = {
        "pytorch": FillMaskPipeline,
        "onnx": OnnxFillMaskPipeline,
    }

    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForMaskedLM.from_pretrained(model)

    return pipelines[backend](model=model, tokenizer=tokenizer)


def test_file_mask():
    from utils import model_util

    ray.init()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    gptvm_lazy_import()

    # get current dir
    current_dir = os.getcwd()
    model_dir = current_dir + "/model_data"
    model_name = "bert"
    # download the mnist model
    model_util.dload_model(model_name, model_dir)

    classifier = pipeline("onnx", model=model_dir + "/" + model_name)

    output = classifier("Paris is the [MASK] of France.")

    print(output)
    # output must have 'capital'
    assert output[0]["token_str"] == "capital"
    ray.shutdown()
