from transformers import FillMaskPipeline, AutoTokenizer, AutoModelForMaskedLM

from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel

import torch
import struct
import gptvm
from urllib import parse
import numpy


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
        self.task = gptvm.GVTask.create("onnx/model.onnx", gptvm.GV_NV_GPU)

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
