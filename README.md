# 1. Eval

运行时请在config.py中执行本地的model_path/huggingface中的模型名称 （模型起vllm的时候没有制定具体的模型名称）

如果服务器无法连接外网，请首先运行scieval/offline_download.py文件，并在其中指定数据集的名称; 在对应benchmark的 _load_from_huggingface中支持本地读取

注意：
1. 如测的新模型不属于任何一个模型series，需要写一个该模型的类（继承自BaseModel）
2. 每一个新Benchmark需要新创建一个类，对其中的build_prompt()函数好好设计

已支持本地load的benchmark & 本地load model：
- B: Chembench
- M: Qwen3-8B, Qwen3-4B-Instruct

注意：
每个benchmark的max_new_token可能是不一样的，需要跑一跑估计这个benchmark的max_new_token再做判断



