import torch


def torch_dtype_to_str(torch_dtype):
    return {
        torch.float16: "float16",
        torch.float32: "float32",
        torch.int64: "int64",
        torch.int32: "int32",
        torch.int8: "int8",
        torch.bool: "bool",
        torch.bfloat16: "bfloat16",
    }[torch_dtype]
