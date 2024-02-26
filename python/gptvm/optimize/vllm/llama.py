from gptvm.injector import register_method_injector
from gptvm import log
from ..utils import torch_dtype_to_str
from transformers import LlamaForCausalLM
from vllm import LLM, SamplingParams
import torch


@register_method_injector(LlamaForCausalLM, "from_pretrained")
def from_pretrained(
    cls,
    pretrained_model_name_or_path,
    # *model_args,
    config=None,
    cache_dir=None,
    ignore_mismatched_sizes=False,
    force_download=False,
    local_files_only=False,
    token=None,
    revision="main",
    use_safetensors=None,
    **kwargs
):
    kwargs["device_map"] = "cpu"
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        # *model_args,
        config=config,
        cache_dir=cache_dir,
        ignore_mismatched_sizes=ignore_mismatched_sizes,
        force_download=force_download,
        local_files_only=local_files_only,
        token=token,
        revision=revision,
        use_safetensors=use_safetensors,
        **kwargs
    )
    dtype = torch_dtype_to_str(model.config.torch_dtype)
    model.vllm = LLM(
        model=pretrained_model_name_or_path, dtype=dtype, enforce_eager=False
    )
    return model


@register_method_injector(LlamaForCausalLM, "generate")
def generate(
    self,
    inputs=None,
    generation_config=None,
    logits_processor=None,
    stopping_criteria=None,
    prefix_allowed_tokens_fn=None,
    synced_gpus=None,
    assistant_model=None,
    streamer=None,
    negative_prompt_ids=None,
    negative_prompt_attention_mask=None,
    **kwargs
):
    if inputs is None:
        inputs = kwargs["input_ids"]
    
    output_len = kwargs["max_new_tokens"]

    sampling_params = SamplingParams(n=1, temperature=0, max_tokens=output_len)
    result = self.vllm.generate(
        sampling_params=sampling_params, prompt_token_ids=inputs.tolist()
    )
    if streamer is not None:
        for token_id in result[0].outputs[0].token_ids:
            streamer.put(torch.as_tensor([token_id]))
        streamer.end()
        return

    return torch.IntTensor([out.prompt_token_ids + out.outputs[0].token_ids for out in result])
