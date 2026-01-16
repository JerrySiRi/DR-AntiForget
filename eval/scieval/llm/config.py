from functools import partial

from .qwen3 import Qwen3LLM


qwen3llm_series = {
    "Qwen3-8B": partial(
        Qwen3LLM,
        model_path="Qwen/Qwen3-8B",
        use_vllm=True,
        temperature=0.7,
        max_new_tokens=4096,
    ),
    "Qwen3-8B-Instruct": partial(
        Qwen3LLM,
        model_path="Qwen/Qwen3-8B-Instruct",
        use_vllm=True,
        temperature=0.7,
        max_new_tokens=4096,
    ),
}
