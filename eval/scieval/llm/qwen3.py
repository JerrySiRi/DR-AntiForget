import torch

from ..vlm.base import BaseModel


class Qwen3LLM(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = False
    allowed_types = ['text']

    def __init__(
        self,
        model_path: str = "", # need designate in config.py
        #! your local path, or huggingface name
        use_vllm: bool = True,
        max_model_len: int = 8192,
        max_num_seqs: int = 8,
        gpu_memory_utilization: float = 0.90,
        temperature: float = 0,
        top_p: float = 0.8,
        top_k: int = 20,
        max_tokens: int = 1024, #* default
        stop: list[str] | None = None,
        trust_remote_code: bool = True,
        dtype: str = "auto",
        tensor_parallel_size: int | None = None,
        **kwargs,
    ):
        super().__init__()

        self.model_path = model_path
        self.use_vllm = use_vllm

        self.generation_defaults = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens,
        }
        if stop:
            self.generation_defaults["stop"] = stop

        if self.use_vllm:
            from vllm import LLM

            gpu_count = torch.cuda.device_count()
            tp_size = tensor_parallel_size
            if tp_size is None:
                tp_size = max(1, gpu_count)

            llm_kwargs = {
                "model": model_path,
                "tensor_parallel_size": tp_size,
                "max_num_seqs": max_num_seqs,
                "gpu_memory_utilization": gpu_memory_utilization,
                "dtype": dtype,
                # "max_model_len": max_model_len,
            }
            self.llm = LLM(**llm_kwargs)
    
            self._backend = "vllm"
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
                torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else None),
                device_map="auto" if torch.cuda.is_available() else None,
            )
            self._backend = "transformers"

    def build_prompt(self, line, dataset=None):
        raise NotImplementedError

    def _message_to_prompt(self, message):
        texts = [x["value"] for x in message if x["type"] == "text"]
        return "\n".join(texts)

    def generate_inner(self, message, dataset=None):
        prompt = self._message_to_prompt(message)

        if self._backend == "vllm":
            from vllm import SamplingParams

            sampling_params = SamplingParams(**self.generation_defaults)
            outputs = self.llm.generate(prompt, sampling_params=sampling_params)
            if not outputs:
                return ""
            return outputs[0].outputs[0].text

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.generation_defaults["max_tokens"],
            do_sample=(self.generation_defaults.get("temperature", 0) > 0),
            temperature=self.generation_defaults.get("temperature", 0.0),
            top_p=self.generation_defaults.get("top_p", 1.0),
            top_k=self.generation_defaults.get("top_k", 0),
        )
        gen = out[0, inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)
