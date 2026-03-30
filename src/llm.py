"""LLM API client wrapper with retry and token tracking."""

import time
import logging
from dataclasses import dataclass, field
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


@dataclass
class TokenTracker:
    total_input: int = 0
    total_output: int = 0
    total_calls: int = 0
    call_log: list = field(default_factory=list)

    def record(self, resp: LLMResponse, label: str = "") -> None:
        self.total_input += resp.input_tokens
        self.total_output += resp.output_tokens
        self.total_calls += 1
        if label:
            self.call_log.append({
                "label": label,
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
            })

    def summary(self) -> dict:
        return {
            "total_input_tokens": self.total_input,
            "total_output_tokens": self.total_output,
            "total_tokens": self.total_input + self.total_output,
            "total_calls": self.total_calls,
        }

    def reset(self) -> None:
        self.total_input = 0
        self.total_output = 0
        self.total_calls = 0
        self.call_log.clear()


class LLMClient:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        max_retries: int = 3,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.tracker = TokenTracker()
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=120.0)

    def chat(
        self,
        messages: list[dict],
        stop: list[str] | None = None,
        label: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send chat completion request with retry."""
        temp = temperature if temperature is not None else self.temperature
        mt = max_tokens if max_tokens is not None else self.max_tokens

        for attempt in range(self.max_retries):
            try:
                t0 = time.time()
                # Disable thinking mode for reasoning models (e.g., Qwen3.5)
                extra = {}
                if "qwen3" in self.model.lower() or "qwq" in self.model.lower():
                    extra["extra_body"] = {"enable_thinking": False}

                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=mt,
                    stop=stop,
                    **extra,
                )
                latency = (time.time() - t0) * 1000

                content = resp.choices[0].message.content or ""
                usage = resp.usage
                result = LLMResponse(
                    content=content.strip(),
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                    latency_ms=latency,
                )
                self.tracker.record(result, label=label)
                return result

            except Exception as e:
                wait = 2 ** (attempt + 1)
                logger.warning(f"LLM call failed (attempt {attempt+1}/{self.max_retries}): {e}. Retrying in {wait}s...")
                if attempt < self.max_retries - 1:
                    time.sleep(wait)
                else:
                    logger.error(f"LLM call failed after {self.max_retries} attempts")
                    raise

    def complete_text(
        self,
        prompt: str,
        stop: list[str] | None = None,
        label: str = "",
        system: str | None = None,
    ) -> str:
        """Convenience: single prompt string → response text."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, stop=stop, label=label).content
