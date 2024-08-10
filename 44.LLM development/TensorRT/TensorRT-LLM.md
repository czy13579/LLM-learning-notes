## TensorRT-LLM作用
第一、**大模型参数量大，推理成本高**。以10B参数规模的大模型为例，使用FP16数据类型进行部署至少需要20GB以上（模型权重+KV缓存等）。

第二、纯TensorRT使用较复杂，**ONNX存在内存限制**。深度学习模型通常使用各种框架（如PyTorch、TensorFlow、Keras等）进行训练和部署，而每个框架都有自己的模型表示和存储格式。因此，开发者通常使用 ONNX 解决深度学习模型在不同框架之间的互操作性问题。比如：TensorRT 就需要先将 PyTorch 模型转成 ONNX，然后再将 ONNX 转成 TensorRT。

第三、 纯**FastTransformer**使用门槛高。FastTransformer 是用 **C++ 实现**的；同时，它的接口和文档相对较少，用户可能需要更深入地了解其底层实现和使用方式，这对于初学者来说可能会增加学习和使用的难度。

TensorRT-LLM 为用户提供了**易于使用的 Python API** 来定义大语言模型 (LLM) 并构建 TensorRT 引擎，以便在 NVIDIA GPU 上高效地执行推理。 TensorRT-LLM 还包含用于创建执行这些 TensorRT 引擎的 Python 和 C++ 运行时组件。 此外，它还包括一个用于与 NVIDIA Triton 推理服务集成的后端；

关键特性
- 支持多头注意力(Multi-head Attention，MHA)
- 支持多查询注意力 (Multi-query Attention，MQA)
- 支持分组查询注意力(Group-query Attention，GQA)
- 支持飞行批处理（In-flight Batching）
- Paged KV Cache for the Attention
- 支持 张量并行
- 支持 流水线并行
- 支持仅 INT4/INT8 权重量化 (W4A16 & W8A16)
- 支持 SmoothQuant 量化
- 支持 GPTQ 量化
- 支持 AWQ 量化
- 支持 FP8
- 支持贪心搜索（Greedy-search）
- 支持波束搜索（Beam-search）
- 支持旋转位置编码（RoPE）