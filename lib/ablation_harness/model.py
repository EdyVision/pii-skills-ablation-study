"""ModelInference — unified interface for MLX and Transformers backends."""

from __future__ import annotations


class ModelInference:
    """Unified interface for MLX and Transformers inference."""

    def __init__(self, model_name: str, hardware: str, model_config: dict):
        self.model_name = model_name
        self.hardware = hardware
        self.model_config = model_config
        self.model = None
        self.tokenizer = None

    def load(self):
        """Load model based on hardware."""
        if self.hardware == "mlx":
            self._load_mlx(self.model_config["mlx"])
        else:
            self._load_transformers(self.model_config["hf"])

    def unload(self):
        """Drop model and tokenizer so the next model can load without OOM."""
        self.model = None
        self.tokenizer = None
        if hasattr(self, "_generate_fn"):
            del self._generate_fn

    def _load_mlx(self, model_id: str):
        """Load model with MLX."""
        from mlx_lm import load

        print(f"Loading MLX model: {model_id}")
        self.model, self.tokenizer = load(model_id)
        print("Model loaded successfully")

    def _load_transformers(self, model_id: str):
        """Load model with Transformers."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        print(f"Loading Transformers model: {model_id}")

        device = "cuda" if self.hardware == "cuda" else "cpu"
        dtype = torch.float16 if self.hardware == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Flash Attention 2: uses fused, tiled CUDA kernels for the attention
        # computation. Doesn't change model outputs — just reduces VRAM and
        # wall-clock time. Applied uniformly so it's not an ablation confound.
        # Requires flash-attn package (only installable on CUDA machines).
        attn_impl = None
        if self.hardware == "cuda":
            try:
                import flash_attn  # noqa: F401

                attn_impl = "flash_attention_2"
                print("Flash Attention 2 available, enabling")
            except ImportError:
                print("flash-attn not installed, using default attention")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="auto" if self.hardware == "cuda" else None,
            attn_implementation=attn_impl,
        )
        if self.hardware == "cpu":
            self.model = self.model.to(device)

        if self.hardware == "cuda":
            self.model = torch.compile(self.model, mode="default")
            print(f"Model compiled with torch.compile on {device}")
        else:
            print(f"Model loaded on {device}")

    def _format_prompt(self, prompt: str) -> str:
        """Wrap user prompt in the model's chat template."""
        if self.tokenizer is None:
            return prompt
        try:
            if hasattr(self.tokenizer, "apply_chat_template"):
                messages = [{"role": "user", "content": prompt}]
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        except Exception:
            pass
        return prompt

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate response from prompt."""
        formatted = self._format_prompt(prompt)
        if self.hardware == "mlx":
            return self._generate_mlx(formatted, max_tokens)
        else:
            return self._generate_transformers(formatted, max_tokens)

    def generate_from_messages(
        self, messages: list[dict], max_tokens: int = 1024
    ) -> str:
        """Generate from a full message list (system, user, assistant, tool, ...) so the model sees multi-turn structure. Used for Llama/Qwen native tool-calling path."""
        formatted = self._format_messages(messages)
        if formatted is None:
            return ""
        if self.hardware == "mlx":
            return self._generate_mlx(formatted, max_tokens)
        return self._generate_transformers(formatted, max_tokens)

    def _format_messages(self, messages: list[dict]) -> str | None:
        """Format message list for chat template. Normalizes role+content; maps tool to user if template does not support tool role."""
        if not messages or self.tokenizer is None:
            return None
        try:
            if not hasattr(self.tokenizer, "apply_chat_template"):
                return None
            normalized = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content") or ""
                if role == "tool":
                    name = m.get("name", "tool")
                    normalized.append(
                        {"role": "user", "content": f"Tool result ({name}):\n{content}"}
                    )
                else:
                    normalized.append({"role": role, "content": content})
            return self.tokenizer.apply_chat_template(
                normalized,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return None

    def _generate_mlx(self, prompt: str, max_tokens: int) -> str:
        """Generate with MLX."""
        from mlx_lm import generate

        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        return response

    def _generate_transformers(self, prompt: str, max_tokens: int) -> str:
        """Generate with Transformers."""
        import torch
        from transformers import GenerationConfig

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.hardware == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        gen_config = GenerationConfig(
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        with torch.no_grad():
            outputs = self.model.generate(**inputs, generation_config=gen_config)

        new_tokens = outputs[0][input_length:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate_batch(self, prompts: list[str], max_tokens: int = 1024) -> list[str]:
        """Generate responses for a list of prompts (single-turn)."""
        if not prompts:
            return []
        formatted = [self._format_prompt(p) for p in prompts]
        if self.hardware == "mlx":
            return self._generate_batch_mlx(formatted, max_tokens)
        return self._generate_batch_transformers(formatted, max_tokens)

    def _generate_batch_mlx(self, prompts: list[str], max_tokens: int) -> list[str]:
        try:
            from mlx_lm import batch_generate

            # mlx-lm batch_generate expects pre-tokenized prompts (List[List[int]])
            tokenized = [
                self.tokenizer.encode(p, add_special_tokens=True) for p in prompts
            ]
            result = batch_generate(
                self.model,
                self.tokenizer,
                prompts=tokenized,
                max_tokens=max_tokens,
                verbose=False,
            )
            # Extract text from BatchResponse
            if hasattr(result, "texts"):
                return result.texts
            if isinstance(result, list):
                return result
            return list(result)
        except Exception as e:
            if not getattr(self, "_mlx_batch_fallback_warned", False):
                print(
                    f"MLX: batch_generate failed ({e}), using sequential (effective batch 1).",
                    flush=True,
                )
                self._mlx_batch_fallback_warned = True
            return [self._generate_mlx(p, max_tokens) for p in prompts]

    def _generate_batch_transformers(
        self, prompts: list[str], max_tokens: int
    ) -> list[str]:
        import torch
        from transformers import GenerationConfig

        padding_side = getattr(self.tokenizer, "padding_side", "right")
        self.tokenizer.padding_side = "left"
        try:
            inputs = self.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True
            )
            if self.hardware == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            input_lengths = inputs["attention_mask"].sum(dim=1)
            gen_config = GenerationConfig(
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            with torch.no_grad():
                outputs = self.model.generate(**inputs, generation_config=gen_config)
            out = []
            for i in range(len(prompts)):
                new_tokens = outputs[i][input_lengths[i] :]
                out.append(
                    self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                )
            return out
        finally:
            self.tokenizer.padding_side = padding_side
