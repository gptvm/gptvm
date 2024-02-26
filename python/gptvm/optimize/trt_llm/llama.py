from gptvm.injector import register_method_injector
from gptvm import log

from .llama_weight import load_from_hf_llama
from ..utils import torch_dtype_to_str

from transformers import LlamaForCausalLM
import torch
import tensorrt_llm
import tensorrt
from tensorrt_llm.builder import Builder
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.network import net_guard
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.generation import ModelConfig, GenerationSession
from tensorrt_llm.runtime.model_runner import ModelRunner


MODEL_NAME = "llama"
MAX_BATCH_SIZE = 8
MAX_INPUT_LEN = 2048
MAX_OUTPUT_LEN = 1024
MAX_BEAM_WIDTH = 1


def trt_build_llama(hf_model):
    log.debug("Building llama model using TensorRT LLM")
    hf_config = hf_model.config
    dtype = torch_dtype_to_str(hf_config.torch_dtype)
    inter_size = hf_config.intermediate_size
    n_embd = hf_config.hidden_size
    n_head = hf_config.num_attention_heads
    n_layer = hf_config.num_hidden_layers
    n_positions = hf_config.max_position_embeddings
    n_kv_head = None
    if hasattr(hf_config, "num_key_value_heads"):
        n_kv_head = hf_config.num_key_value_heads
    vocab_size = hf_config.vocab_size
    hidden_act = hf_config.hidden_act
    rms_norm_eps = hf_config.rms_norm_eps

    model = tensorrt_llm.models.LLaMAForCausalLM(
        dtype=dtype,
        num_layers=n_layer,
        num_heads=n_head,
        num_kv_heads=n_kv_head,
        hidden_size=n_embd,
        vocab_size=vocab_size,
        hidden_act=hidden_act,
        max_position_embeddings=n_positions,
        mlp_hidden_size=inter_size,
        rms_norm_eps=rms_norm_eps,
    )

    load_from_hf_llama(model, hf_model, dtype=dtype)

    builder = Builder()
    network = builder.create_network()
    network.trt_network.name = MODEL_NAME
    network.plugin_config.set_gpt_attention_plugin(dtype=dtype)
    network.plugin_config.set_gemm_plugin(dtype=dtype)
    network.plugin_config.set_context_fmha(ContextFMHAType.enabled)
    network.plugin_config.enable_remove_input_padding()

    with net_guard(network):
        # Prepare
        network.set_named_parameters(model.named_parameters())

        # Forward
        inputs = model.prepare_inputs(
            MAX_BATCH_SIZE, MAX_INPUT_LEN, MAX_OUTPUT_LEN, True, MAX_BEAM_WIDTH
        )
        model(*inputs)

    tensorrt_llm.graph_rewriting.optimize(network)

    # BuilderConfig \
    #       Network -> Engine \
    #             ModelConfig -> GenerationSession -> ModelRunner
    builder_config = builder.create_builder_config(
        name=MODEL_NAME,
        precision=dtype,
        num_layers=n_layer,
        num_heads=n_head,
        num_kv_heads=n_kv_head,
        hidden_size=n_embd,
        vocab_size=vocab_size,
        hidden_act=hidden_act,
        max_position_embeddings=n_positions,
        max_batch_size=MAX_BATCH_SIZE,
        max_input_len=MAX_INPUT_LEN,
        max_output_len=MAX_OUTPUT_LEN,
        max_beam_width=MAX_BEAM_WIDTH,
    )
    engine = builder.build_engine(network, builder_config)
    model_config = ModelConfig(
        hf_config.vocab_size,
        hf_config.num_hidden_layers,
        hf_config.num_attention_heads,
        hf_config.num_attention_heads,
        hf_config.hidden_size,
        gpt_attention_plugin=True,
        remove_input_padding=True,
        model_name=MODEL_NAME,
        dtype=dtype,
    )
    session = GenerationSession(model_config, engine, Mapping())
    runner = ModelRunner(
        session, MAX_BATCH_SIZE, MAX_INPUT_LEN, MAX_OUTPUT_LEN, MAX_BEAM_WIDTH
    )
    return runner


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
    # Redirect hf model to cpu, otherwise we will have two copies of the model on cuda
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
    model.runner = trt_build_llama(model)
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
    if self.runner is None:
        return LlamaForCausalLM.generate(
            self,
            inputs,
            generation_config,
            logits_processor,
            stopping_criteria,
            prefix_allowed_tokens_fn,
            synced_gpus,
            assistant_model,
            streamer,
            negative_prompt_ids,
            negative_prompt_attention_mask,
            **kwargs
        )

    if inputs is None:
        inputs = kwargs["input_ids"]
    # turn batches of inputs into list of tensors which is needed by trt_llm
    inputs = [x.unsqueeze(0) for x in inputs]
    eos_token_id = self.config.eos_token_id
    pad_token_id = self.config.pad_token_id or eos_token_id

    log.debug("Generating from TensorRT LLM llama runner")
    if streamer is not None:
        generators = self.runner.generate(
            inputs,
            streaming=True,
            end_id=eos_token_id,
            pad_id=pad_token_id,
            max_new_tokens=kwargs.get('max_new_tokens', MAX_OUTPUT_LEN),
            return_dict=False,
        ).squeeze(1)
        input_len = inputs.size(1)
        for i, outputs in enumerate(generators):
            streamer.put(outputs[0][:, input_len + i])

        streamer.end()
        return

    return self.runner.generate(
        inputs,
        end_id=eos_token_id,
        pad_id=pad_token_id,
        max_new_tokens=kwargs.get('max_new_tokens', MAX_OUTPUT_LEN),
        return_dict=False,
    ).squeeze(1)
