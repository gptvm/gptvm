from collections import OrderedDict

from gptvm.injector import register_method_injector
from gptvm import log

from .bert_weight import load_from_hf_bert, load_from_hf_qa_bert

from transformers import BertModel,BertConfig, BertForQuestionAnswering
from transformers.modeling_outputs import QuestionAnsweringModelOutput
import torch
import tensorrt_llm
import tensorrt as trt
from tensorrt_llm.builder import Builder
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.network import net_guard
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime import Session, TensorInfo

MODEL_NAME = "bert"
MAX_BATCH_SIZE = 256
MAX_INPUT_LEN = 512
TORCH_DTYPE=torch.float16

def torch_dtype_to_trt_dtype(torch_dtype):
    return {
        torch.float16: trt.float16,
        torch.float32: trt.float32,
        torch.int64: trt.int64,
        torch.int32: trt.int32,
        torch.int8: trt.int8,
        torch.bool: trt.bool,
        torch.bfloat16: trt.float16,
    }[torch_dtype]

def trt_dtype_to_torch(dtype):
    if dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    elif dtype == trt.int32:
        return torch.int32
    else:
        raise TypeError("%s is not supported" % dtype)


def trt_build_bert(hf_model):
    log.debug("Building  model using TensorRT LLM")
    hf_config = hf_model.config
    trt_dtype = torch_dtype_to_trt_dtype(TORCH_DTYPE)
    model = tensorrt_llm.models.BertForQuestionAnswering(
        num_layers=hf_config.num_hidden_layers,
        num_heads=hf_config.num_attention_heads,
        hidden_size=hf_config.hidden_size,
        vocab_size=hf_config.vocab_size,
        hidden_act=hf_config.hidden_act,
        max_position_embeddings=hf_config.max_position_embeddings,
        type_vocab_size=hf_config.type_vocab_size,
        num_labels=2,
        #mapping=Mapping(world_size=args.world_size,
        #                rank=args.rank,
        #                tp_size=args.world_size),  # TP only
        dtype=trt_dtype)

    bert_config = BertConfig(
        vocab_size=hf_config.vocab_size,
        hidden_size=hf_config.hidden_size,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_attention_heads=hf_config.num_attention_heads,
        intermediate_size=4 * hf_config.hidden_size,
        hidden_act=hf_config.hidden_act,
        max_position_embeddings=hf_config.max_position_embeddings,
        torch_dtype=TORCH_DTYPE,
    )

    load_from_hf_qa_bert(model, hf_model, bert_config, fp16=(trt_dtype == trt.float16))

    builder = Builder()
    network = builder.create_network()
    network.trt_network.name = MODEL_NAME

    bs_range = [1, (MAX_BATCH_SIZE + 1) // 2, MAX_BATCH_SIZE]
    inlen_range = [1, (MAX_INPUT_LEN + 1) // 2, MAX_INPUT_LEN]
    with net_guard(network):
        # Prepare
        network.set_named_parameters(model.named_parameters())

        # Forward
        input_ids = tensorrt_llm.Tensor(
            name='input_ids',
            dtype=trt.int32,
            shape=[-1, -1],
            dim_range=OrderedDict([('batch_size', [bs_range]),
                                   ('input_len', [inlen_range])]),
        )

        # also called segment_ids
        token_type_ids = tensorrt_llm.Tensor(
            name='token_type_ids',
            dtype=trt.int32,
            shape=[-1, -1],
            dim_range=OrderedDict([('batch_size', [bs_range]),
                                   ('input_len', [inlen_range])]),
        )

        input_lengths = tensorrt_llm.Tensor(name='input_lengths',
                                            dtype=trt.int32,
                                            shape=[-1],
                                            dim_range=OrderedDict([
                                                ('batch_size', [bs_range])
                                            ]))

        # logits for QA BERT, or hidden_state for vanila BERT
        output = model(input_ids=input_ids,
                        input_lengths=input_lengths,
                        token_type_ids=token_type_ids)

        output_name = 'logits'
        # Mark outputs
        output_dtype = trt_dtype
        output.mark_output(output_name, output_dtype)


    # BuilderConfig \
    #       Network -> Engine \
    #             ModelConfig -> GenerationSession -> ModelRunner
    builder_config = builder.create_builder_config(
        name=MODEL_NAME,
        precision='float16',
        timing_cache='model.cache',
        tensor_parallel=1,
        max_batch_size=MAX_BATCH_SIZE,
        max_input_len=MAX_INPUT_LEN,
    )
    engine = builder.build_engine(network, builder_config)
    session = Session.from_serialized_engine(engine)
    return session

@register_method_injector(BertForQuestionAnswering, "from_pretrained")
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
    model = BertForQuestionAnswering.from_pretrained(
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
    # the return value of BertModel Session
    model.runner = trt_build_bert(model)
    return model

@register_method_injector(BertForQuestionAnswering, "forward")
def forward(
    self,
    inputs_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    start_positions=None,
    end_positions=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    if self.runner is None:
        return BertForQuestionAnswering.forward(
            self,
            inputs_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            start_positions,
            end_positions,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

    batch_size = inputs_ids.shape[0]
    seq_len = inputs_ids.shape[1]
    input_lengths = seq_len * torch.ones(
            (batch_size, ), dtype=torch.int32)
    log.debug("Step in hooked forward")
    inputs = {
        'input_ids': inputs_ids,
        'input_lengths': input_lengths,
        'token_type_ids': token_type_ids
    }

    # put inputs on gpu
    for k, v in inputs.items():
        inputs[k] = v.int().cuda()

    output_info = self.runner.infer_shapes([
        TensorInfo('input_ids', trt.DataType.INT32, inputs_ids.shape),
        TensorInfo('input_lengths', trt.DataType.INT32,
                   input_lengths.shape),
        TensorInfo('token_type_ids', trt.DataType.INT32,
                   token_type_ids.shape),
    ])
    outputs = {
        t.name: torch.empty(tuple(t.shape),
                            dtype=trt_dtype_to_torch(t.dtype),
                            device='cuda')
        for t in output_info
    }

    output_name = 'logits'
    assert output_name in outputs, f'{output_name} not found in outputs, check if build.py set the name correctly'
    log.debug("calling session.run()")
    stream = torch.cuda.current_stream().cuda_stream
    ok = self.runner.run(inputs, outputs, stream)
    assert ok, " Bert runtime execution failed"
    torch.cuda.synchronize()
    logits = outputs[output_name]
    # below is copied from BertForQuestionAnswering.forward
    start_logits, end_logits = logits.split(1, dim=-1)
    start_logits = start_logits.squeeze(-1).contiguous()
    end_logits = end_logits.squeeze(-1).contiguous()
    total_loss = None
    if start_positions is not None and end_positions is not None:
        # If we are on multi-GPU, split add a dimension
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)
        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        ignored_index = start_logits.size(1)
        start_positions = start_positions.clamp(0, ignored_index)
        end_positions = end_positions.clamp(0, ignored_index)

        loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2

    #if not return_dict:
    #    output = (start_logits, end_logits) + outputs[2:]
    #    return ((total_loss,) + output) if total_loss is not None else output

    return QuestionAnsweringModelOutput(
        loss=total_loss,
        start_logits=start_logits,
        end_logits=end_logits,
        hidden_states=None,
        attentions=None,
        )
