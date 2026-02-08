"""
Custom Haystack Invocation Layer for ctransformers.
Provides LlamaCPPInvocationLayer to use GGUF models with Haystack's PromptNode.
Uses ctransformers as the backend instead of llama-cpp-python.
"""

import os
import logging
from typing import Dict, List, Optional, Union

from haystack.nodes.prompt.invocation_layer import PromptModelInvocationLayer

logger = logging.getLogger(__name__)


class LlamaCPPInvocationLayer(PromptModelInvocationLayer):
    """
    A custom invocation layer that uses ctransformers to run
    GGUF-format LLM models locally on CPU.
    """

    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = 512,
        max_context: int = 2048,
        temperature: float = 0.75,
        top_p: float = 1.0,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        threads: Optional[int] = None,
        **kwargs,
    ):
        if not model_name_or_path or len(model_name_or_path) == 0:
            raise ValueError("model_name_or_path must not be None or empty string")

        super().__init__(model_name_or_path, **kwargs)

        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.max_context = max_context
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repeat_penalty = repeat_penalty
        self.threads = threads or os.cpu_count()

        try:
            from ctransformers import AutoModelForCausalLM
        except ImportError:
            raise ImportError(
                "Could not import ctransformers python package. "
                "Please install it with `pip install ctransformers`."
            )

        # Determine model directory and file
        if os.path.isfile(model_name_or_path):
            model_dir = os.path.dirname(model_name_or_path) or "."
            model_file = os.path.basename(model_name_or_path)
        else:
            model_dir = "."
            model_file = model_name_or_path

        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            model_file=model_file,
            model_type="llama",
            context_length=self.max_context,
            threads=self.threads,
        )

    def _ensure_token_limit(
        self, prompt: Union[str, List[Dict[str, str]]]
    ) -> Union[str, List[Dict[str, str]]]:
        """Ensure the prompt does not exceed the context limit."""
        if isinstance(prompt, str):
            return prompt[: self.max_context]
        return prompt

    def invoke(self, *args, **kwargs):
        """
        Invoke the LLM model to generate text based on the given prompt.
        """
        prompt = kwargs.get("prompt", args[0] if args else "")

        generated_text = self.model(
            prompt,
            max_new_tokens=self.max_length,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repeat_penalty,
        )

        return [generated_text]

    @classmethod
    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        """
        Check if this invocation layer supports the given model.
        Returns True for any non-empty model path (assumes GGUF format).
        """
        return model_name_or_path is not None and len(model_name_or_path) > 0
