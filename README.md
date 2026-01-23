# Quick Start

环境维护均适用uv，如对虚拟环境进行更新，请保证push之前运行uv sync来同步uv.lock


# 1. Eval

## 1.1 scievalkit

运行时请在config.py中执行本地的model_path/huggingface中的模型名称 （模型起vllm的时候没有制定具体的模型名称）

如果服务器无法连接外网，请首先运行scieval/offline_download.py文件，并在其中指定数据集的名称; 在对应benchmark的 _load_from_huggingface中支持本地读取

注意：
- 如测的新模型不属于任何一个模型series，需要写一个该模型的类（继承自BaseModel）
- 每一个新Benchmark需要新创建一个类，对其中的build_prompt()函数好好设计 & 参考evalscope中公认的评测脚本（尤其是抽取和规范化流程）
- 已支持本地load的benchmark & 本地load model：


注意： 
- 每个benchmark的超参数配置是不一样的，需要重新配置config.py文件中的模型超参数（按evalscope，或其他公认的评测脚本来）
- 需要把evalscope的脚本直接放过来，不能直接调evalscope的代码，会有严重的依赖冲突

## 1.2 Evalscope

请详见google docs中注意事项


# 2. SFT

多次分发后真正运行的代码在./sft/LlamaFactory/src/llamafactory/train，里边可以做train & evaluation & predict，说不定可以和Scieval和在一起？

DeepSpeed 在初始化时要检查/编译 CUDA Op（含nvcc），在运行前需要保证安装CUDA Toolkit, 环境变量CUDA_HOME被设置等等（CityUHK服务器需要做module load cuda/12.4）
