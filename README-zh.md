# GPTVM是什么

GPTVM是一个针对生成式预训练Transformer（GPT）类模型的虚拟机（VM）架构。以大语言模型的生成式服务应用为优化对象，面向从单卡环境到复杂的大规模分布式异构系统的多样性目标平台，GPTVM提供了一个系统级的编译优化和运行时基础设施。

作为一个虚拟机环境，GPTVM提供了一个与模型前端无关，同时也与目标平台后端无关的程序抽象。目前业界主流的大模型推理服务的前端优化和后端优化是完全割裂的实现，大语言模型优化前端框架（vLLM，TGI，DeepSpeed，...）主要在python层实现一些高层的算法和架构优化，比如引入KV cache避免冗余计算，把连续的请求序列重新组织成batch输入以获得更高的GPU资源利用率（continuous batching），多机多卡并行计算（TP，PP）；后端优化主要是在pytorch框架或推理引擎中做一些模型和算子层的优化，比如融合算子（flash attention），低比特量化等。但这种方式不能形成一个针对大语言模型推理的系统级解决方案，具有以下缺点：1）前端优化经常需要针对每个模型做大量手写优化的实现；2）前端优化扩展性差，添加新的优化方法会和已有代码产生高度耦合；3）后端优化无法扩展到到其它平台，特别是大规模异构分布式的算力平台上。

为了解决这些问题，我们需要在前后端优化中间提供一层额外的抽象层次来弥补高层大语言模型框架和后端模型和算子实现中间的语义差异。把传统程序语言虚拟机中的编译和运行时优化的思想带到大语言模型的优化场景中，虚拟机架构的优势在于：

* 平台无关 - 一次编译，各处运行，包括运行在高度复杂的分布式异构系统上。
* 模型无关 - 通用优化，无需针对各个模型做手写优化实现。
* 编译优化 - 把用户的高层API编译成GPTVM原语，并基于这一层中间表示做全程序优化。
* 托管运行时 - 精细控制程序的运行时行为并提供自动存储资源管理。

# 关键特性

作为一个基于编译技术的、面向LLM Serving和Fine-tuning训练场景的分布式计算平台，GPTVM具有以下特性：

* 多后端支持

  GPTVM提供一个平台无关的虚拟机架构，可以支持大语言模型应用部署到各种算力硬件上。GPTVM虚拟机定义了一组LLM原语，只要后端实现了这些原语，就可以自然支持所有大语言模型在这种加速器硬件上的部署。这使各种大语言模型应用可以轻松的部署到非NVIDIA的GPU设备上。

* 支持分布式系统

  GPTVM的架构设计包含了对大规模分布式系统作为目标平台的考虑。GPTVM运行时系统的核心是一个基于actor的执行引擎，以保证大语言模型应用可以在大规模分布式系统上获得很好的可扩展性。

* 自动编译和运行时优化

  GPTVM的优化是以所有大语言模型和应用为目标，通过一个定义清晰的程序中间表示，编译器和运行时系统可以更容易的理解程序的语义，以模型无关并且平台无关的方式完成优化，避免为每个模型和每个目标平台编写手工优化代码。同时GPTVM的自动优化可以做到与主流大语言模型优化框架相当的性能水平。

* 自动并行

  GPTVM可以针对特定的模型结构输入，在特定的目标平台配置下，自动选择最优的并行策略。GPTVM编译器会分析模型的计算和访存模式，把大模型的计算进行拆解成可以在单卡上运行的子任务，以满足计算和存储资源的限制。GPTVM的运行时系统负责把这些子任务在整个分布式异构系统上做调度并自动管理存储资源和数据传输。

# 从源码编译GPTVM

## 依赖包

Ubuntu 22.04系统下需要以下安装包。

```shell
apt install build-essential python3 python3-dev cmake protobuf-compiler libjsoncpp-dev libjsoncpp25
```

## 更新submodule

```shell
git submodule update --init
```

## 使用cmake编译

在源码目录下创建build目录，并运行cmake。

```shell
cmake .. && make
```
或使用ninja编译。
```shell
cmake -G Ninja .. && ninja
```

## 安装

要创建可以部署的安装包，在cmake编译时提供 `-DCMAKE_INSTALL_PREFIX=<install_dir>` 配置选项，然后运行
`make install` 或 `ninja install`。

# 使用GPTVM

设置 `PYTHONPATH` 环境变量。
```shell
export PYTHONPATH=$PYTHONPATH:$PWD/gptvm/python:$PWD/../python
```

设置 `PATH` 环境变量。
```shell
export PATH=$PATH:$PWD/bin
```

使用 `gptvm` 命令行工具代替 `python` 运行python应用程序。
```shell
gptvm <your_application.py>
```

运行大语言模型示例。
+ 下载LLaMA模型并准备Python脚本llama.py:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, GenerationConfig
import torch

model = AutoModelForCausalLM.from_pretrained("<LLaMA模型文件所在目录>", torch_dtype=torch.float32, device_map='cpu', _attn_implementation='eager')
tokenizer = AutoTokenizer.from_pretrained("<LLaMA模型文件所在目录>")
prompt = "Hello there! How are you doing?"
inputs = tokenizer(prompt, return_tensors="pt")
generate_ids = model.generate(inputs.input_ids, max_length=512, max_new_tokens=512)
output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output)
```
+ 指定优化策略运行:
```shell
gptvm  -d --opt=torch llama.py
```
