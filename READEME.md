# 1. Eval

运行时请在config.py中执行本地的model_path/huggingface中的模型名称 （模型起vllm的时候没有制定具体的模型名称）

注意：
1. 如测的新模型不属于任何一个模型series，需要写一个该模型的类（继承自BaseModel）
2. 每一个新Benchmark需要新创建一个类，对其中的build_prompt()函数好好设计


