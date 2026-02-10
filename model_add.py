"""
Custom Haystack Invocation Layer for summarization.
Provides LlamaCPPInvocationLayer using Hugging Face Transformers (no GGUF/llama-cpp needed).
Uses a summarization pipeline so the app runs without llama-cpp-python.
"""

import os
import logging
from typing import Dict, List, Optional, Union

from haystack.nodes.prompt.invocation_layer import PromptModelInvocationLayer

logger = logging.getLogger(__name__)


# Default HuggingFace summarization model (runs on CPU, no extra download if already cached)
DEFAULT_SUMMARY_MODEL = "sshleifer/distilbart-cnn-12-6"


class LlamaCPPInvocationLayer(PromptModelInvocationLayer):
    """
    Invocation layer that uses Hugging Face Transformers summarization pipeline.
    Works on CPU without llama-cpp-python or GGUF files.
    """

    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = 512,
        max_context: int = 2048,
        **kwargs,
    ):
        if not model_name_or_path or len(model_name_or_path) == 0:
            model_name_or_path = DEFAULT_SUMMARY_MODEL

        super().__init__(model_name_or_path, **kwargs)

        self.model_name_or_path = model_name_or_path
        self.max_length = min(max_length, 512)  # summarization models often cap at 512
        self.max_context = max_context

        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError(
                "Could not import transformers. "
                "Please install it with `pip install transformers`."
            )

        self.pipeline = pipeline(
            "summarization",
            model=model_name_or_path,
            device=-1,  # CPU
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
        Summarize the given text (prompt) using the Hugging Face model.
        """
        prompt = kwargs.get("prompt", args[0] if args else "")
        if not isinstance(prompt, str):
            prompt = str(prompt)

        # Truncate to max input length for the model (e.g. 1024 for distilbart)
        max_input = 1024
        if len(prompt) > max_input:
            prompt = prompt[:max_input] + "..."

        out = self.pipeline(
            prompt,
            max_length=min(150, self.max_length),
            min_length=30,
            do_sample=False,
        )
        generated_text = out[0]["summary_text"] if out else ""
        return [generated_text]

    @classmethod
    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        return model_name_or_path is not None and len(model_name_or_path) > 0
